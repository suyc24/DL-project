题目：
{QUESTION}

前序步骤：
{PREVIOUS_STEPS}

目标步骤：
{TARGET_STEP}

最终答案：
{FINAL_ANSWER}

请生成 Lean 4 代码，只验证“目标步骤”这个局部 transition。
theorem 名字必须是 `{THEOREM_NAME}`。

要求：
- 只返回一个 Lean 代码块，包含 `import Mathlib`。
- 不使用 `sorry`、`admit`。
- 优先不用 `axiom`，先尝试 `simp`、`ring`、`linarith`、`nlinarith`、`omega` 等。
- 前序步骤可写成局部假设。
- 缺失但必要的前提优先写成局部假设 `h_missing_*`。
- 只有局部假设难以表达时，才允许全局 `axiom obligation_* : ...`。
- 禁止 `axiom ... : False`，也禁止把目标结论直接写成 axiom。
- theorem 不要是 `True` 这种空洞命题。
