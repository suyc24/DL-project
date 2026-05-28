# DeepSeek Pro Single-Step Lean Comparison

Run date: 2026-05-28

Test set: `lean_exp/deepseek_comparison/testset_difficult_3.jsonl`

Problems:
- `OE_TO_maths_en_COMP-1614`: tight subcollections, expected answer `2n-2`
- `OE_TO_maths_en_COMP-1659`: positive-real functional equation, expected answer `f(x)=2x`
- `OE_TO_maths_en_COMP-1678`: exponential divisibility, expected answer `(2,4)`

Settings:
- model: `deepseek-v4-pro`
- problems: 3
- chains per problem: 2
- selected steps per chain: 2
- selected transition attempts per pipeline: 12
- Lean project: `lean_fhis`, Lean 4.29.1

Run directories:
- old pipeline: `lean_single_step/experiments/runs/cmp_old_dspro_20260528_3x2`
- new pipeline: `lean_single_step_formalization/experiments/runs/cmp_new_dspro_20260528_3x2`

## Aggregate Results

| pipeline | Lean files | Lean OK | Lean failed | complete | local `h_missing_*` | global axiom fallback | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `lean_single_step` | 12 | 8 | 4 | n/a | n/a | n/a | verifier accepts `sorry`; several OK files are vacuous or incomplete |
| `lean_single_step_formalization` | 12 | 8 | 4 | 5 OK / 7 total | 3 OK / 4 total | 0 OK / 1 total | bans `sorry`; records dependency mode, but still allows vacuous `True` contracts |

High-quality faithful OK contracts by manual review:
- old pipeline: 3/12 (`1659 c2 s15`, `1659 c2 s9`, `1678 c1 s12`)
- new pipeline: 2/12 strong (`1659 c1 s8`, `1659 c2 s2`), plus 1 partial computation-only OK (`1678 c2 s7`)

Typecheck/semantic gap:
- old pipeline: 5 of 8 typechecked files are semantically hollow or unfaithful because of `sorry`, `True`, or a false/unsupported theorem.
- new pipeline: 5 of 8 typechecked files are semantically hollow or unfaithful because of `True`, direct target-as-hypothesis, or changing the claim.

## Old Pipeline Manual Fidelity Review

| file | Lean | manual fidelity | reason |
|---|---|---|---|
| `OE_TO_maths_en_COMP-1614__1__step11.lean` | OK | unfaithful | NL step is exploratory and realizes a fixed-element construction gives 7, not 8; Lean asserts existence of an 8-set family and proves it by `sorry`. |
| `OE_TO_maths_en_COMP-1614__1__step2.lean` | fail | faithful but invalid Lean | Captures the tightness interpretation reasonably well, but the proof script fails. |
| `OE_TO_maths_en_COMP-1614__2__step1.lean` | OK | vacuous | Restatement step is compiled as `True`; acceptable as a no-op marker, not a real transition contract. |
| `OE_TO_maths_en_COMP-1614__2__step12.lean` | fail | partial | Captures the no-tight contradiction form, but not the incidence-matrix reformulation, and has Finset/Set type errors. |
| `OE_TO_maths_en_COMP-1659__1__step18.lean` | OK | vacuous | NL step is a meta observation; Lean returns `True`. |
| `OE_TO_maths_en_COMP-1659__1__step16.lean` | OK | unfaithful | NL asks whether a lower bound approach might work; Lean asserts a positive uniform lower bound, which is false for `f(x)=2x`, and uses `sorry`. |
| `OE_TO_maths_en_COMP-1659__2__step15.lean` | OK | faithful | Formalizes the proof that `f(t)>t` from positivity, the equation, and the previous no-fixed-point fact. |
| `OE_TO_maths_en_COMP-1659__2__step9.lean` | OK | faithful | Correctly formalizes the algebraic rewrite `z - f(y) + y = z - (f(y)-y)`. |
| `OE_TO_maths_en_COMP-1678__1__step12.lean` | OK | faithful | Correctly verifies `(2,4)` using integer divisibility. |
| `OE_TO_maths_en_COMP-1678__1__step2.lean` | fail | partial | Only tries to prove `7^k - 3^n != 0`; it omits the divisibility-size bound and also gets stuck on Nat subtraction. |
| `OE_TO_maths_en_COMP-1678__2__step18.lean` | OK | unfaithful | Broad no-solution claim is discharged entirely by `sorry` and impossible-looking missing facts. |
| `OE_TO_maths_en_COMP-1678__2__step3.lean` | fail | partial | Attempts the standard divisor-size bound, but does not solve the Nat-to-Int subtraction/divisibility mismatch. |

## New Pipeline Manual Fidelity Review

| file | Lean | dependency mode | manual fidelity | reason |
|---|---|---|---|---|
| `step_contract_OE_TO_maths_en_COMP_1614_c1_s2.lean` | fail | complete | faithful but invalid Lean | Captures the local counting implication: if `x` is in the union and its count is not 1, then the count is at least 2. |
| `step_contract_OE_TO_maths_en_COMP_1614_c1_s11.lean` | OK | complete | unfaithful | The source step is incomplete/confused about singleton subcollections; Lean assumes a `h_singleton_tight` premise and derives contradiction, rather than checking the step. |
| `step_contract_OE_TO_maths_en_COMP_1614_c2_s1.lean` | OK | local missing | vacuous | Restatement becomes `True`, with a meaningless `h_missing_proper_nonempty : ... -> True`. |
| `step_contract_OE_TO_maths_en_COMP_1614_c2_s5.lean` | fail | local missing | partial | Proves only complement nonempty/proper properties, not the full union/intersection and incidence-count translation. |
| `step_contract_OE_TO_maths_en_COMP_1659_c1_s8.lean` | OK | complete | faithful | Correctly formalizes the substitution `x=f(y)` and `x=y`, deriving `f(2f(y)) = f(2y)+2f(y)`. |
| `step_contract_OE_TO_maths_en_COMP_1659_c1_s21.lean` | OK | local missing | unfaithful | Adds `h_missing_linear : forall x, f x = 2*x`, which is essentially the final answer, then proves `True`. |
| `step_contract_OE_TO_maths_en_COMP_1659_c2_s2.lean` | OK | complete | faithful/partial | Correctly formalizes the first consequence of `f(a)=f(b)`: `f(x+a)=f(x+b)`; it does not complete the later injectivity speculation, which is acceptable for a local contract. |
| `step_contract_OE_TO_maths_en_COMP_1659_c2_s3.lean` | OK | local missing | unfaithful | Assumes the desired injectivity as `h_missing_inj_attempt` and proves `True`. |
| `step_contract_OE_TO_maths_en_COMP_1678_c1_s3.lean` | fail | global axiom fallback | useful failure | The NL growth step is not justified locally; Lean exposes missing positivity/size hypotheses and falls back to an axiom. |
| `step_contract_OE_TO_maths_en_COMP_1678_c1_s10.lean` | OK | complete | unfaithful | NL claims the divisor should be positive because "divides" means positive factor; Lean weakens this to `True`, so the semantic error is not caught by typecheck. |
| `step_contract_OE_TO_maths_en_COMP_1678_c2_s1.lean` | fail | complete | useful failure | Faithfully tries to prove `7^k - 3^n > 0` after only excluding equality; this is false because the difference can be negative, and Lean rejects it. |
| `step_contract_OE_TO_maths_en_COMP_1678_c2_s7.lean` | OK | complete | partial/vacuous | Performs the listed `k=2` arithmetic checks, but the theorem conclusion is still `True`, not a precise solution/exclusion statement. |

## Important Caught Errors

1. `1678 c2 s1` in the new pipeline is the clearest valuable example. The natural-language step says that after excluding `7^k = 3^n`, the divisor `d = 7^k - 3^n` is a positive integer. Lean tries to prove `0 < 7^k - 3^n` and fails. This is a real mathematical gap: nonzero does not imply positive.

2. `1678 c1/s3` and old `1678 c2/s3` show the divisor-size argument needs explicit sign/domain choices. The statement `|d| <= k^4+n^2` is natural over integers, but the scripts often formalize `7^k - 3^n` as Nat subtraction, which truncates at zero and changes the claim.

3. `1659 c1 s21` shows a dangerous opposite failure: the new prompt allows `h_missing_*`, and DeepSeek used that to add `h_missing_linear : forall x, f x = 2*x`, i.e. the final theorem, then proved `True`. Lean accepts this shape, but it is semantically invalid as a transition contract.

4. The old pipeline's `sorry` acceptance is a severe confounder: several "OK" files are not proofs at all. In this run, `OE_TO_maths_en_COMP-1614__1__step11.lean`, `OE_TO_maths_en_COMP-1659__1__step16.lean`, and `OE_TO_maths_en_COMP-1678__2__step18.lean` contain `sorry`.

## Conclusion

The newer `lean_single_step_formalization` pipeline is better aligned with the transition-contract goal because it bans `sorry`, records local missing hypotheses, and flags global axioms. However, compiler feedback alone is still not sufficient: DeepSeek can produce Lean that typechecks but is vacuous (`True`), assumes the target/final result as `h_missing_*`, or weakens an incorrect natural-language step into an irrelevant theorem.

For the next iteration, the prompt/checker should reject:
- theorem conclusions syntactically equal to `True` unless the target step is explicitly tagged as a no-op/restatement;
- `h_missing_*` hypotheses that are alpha-equivalent to the target conclusion or final answer;
- unused `h_missing_*` hypotheses;
- proofs containing `sorry` in the old pipeline;
- contracts whose main theorem does not mention the key symbols from the target step.
