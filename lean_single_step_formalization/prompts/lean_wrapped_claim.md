局部命题 wrapped_claim：
```json
{WRAPPED_CLAIM_JSON}
```

请生成 Lean 4 代码，只验证这个 wrapped_claim。
theorem 名字必须是 `{THEOREM_NAME}`。

要求：
- 只返回一个 Lean 代码块，包含 `import Mathlib`。
- theorem 必须表达“premises 推出 conclusion”。
- 只能验证 conclusion 原文声称的命题；不要把 theorem 改成“该命题不成立”、反例存在、或正确版本成立。
- proof_description 只能作为证明思路参考，不能把 conclusion 直接作为 hypothesis。
- 不使用 `sorry`、`admit`。
- 优先不用 `axiom`，先尝试 `simp`、`ring`、`linarith`、`nlinarith`、`omega` 等。
- 缺失但必要的前提优先写成 theorem 的局部假设 `h_missing_*`。
- 对复杂高层标准数学概念，如果 mathlib 里名称不明确或接口成本太高，不要卡住；可以先定义清晰的局部谓词/结构作为接口。
- 如果需要使用高层定理或库里难以找到的接口，优先把它写成 theorem 的局部假设 `h_missing_*`。
- 只有局部接口难以表达时，才允许全局 `axiom obligation_* : ...`；axiom 必须是具体、窄范围、可读的数学接口，不能直接断言最终结论。
- 禁止 `axiom ... : False`，也禁止把目标结论直接写成 axiom。
- theorem 不要是 `True` 这种空洞命题。
- 如果 wrapped_claim 涉及可能为负的整数表达式，优先使用 `ℤ`，不要用 `ℕ` 截断减法偷换语义。
