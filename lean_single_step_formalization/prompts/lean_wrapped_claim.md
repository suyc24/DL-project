局部命题 wrapped_claim：
```json
{WRAPPED_CLAIM_JSON}
```

请生成 Lean 4 代码，只验证这个 wrapped_claim。
theorem 名字必须是 `{THEOREM_NAME}`。

要求：
- 只返回一个 Lean 代码块，包含 `import Mathlib`。
- theorem 必须表达“premises 推出 conclusion”。
- proof_description 只能作为证明思路参考，不能把 conclusion 直接作为 hypothesis。
- 不使用 `sorry`、`admit`。
- 优先不用 `axiom`，先尝试 `simp`、`ring`、`linarith`、`nlinarith`、`omega` 等。
- 缺失但必要的前提优先写成 theorem 的局部假设 `h_missing_*`。
- 只有局部假设难以表达时，才允许全局 `axiom obligation_* : ...`。
- 禁止 `axiom ... : False`，也禁止把目标结论直接写成 axiom。
- theorem 不要是 `True` 这种空洞命题。
- 如果 wrapped_claim 涉及可能为负的整数表达式，优先使用 `ℤ`，不要用 `ℕ` 截断减法偷换语义。
