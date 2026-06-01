你要判断一个数学 CoT 中的目标步骤是否可靠。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "verdict": "valid | invalid | uncertain",
  "issue_type": "none | missing_premise | too_strong | algebra_error | inequality_direction | quantifier_swap | modular_condition | boundary_case | necessity_sufficiency | formalization_issue | other",
  "reason": "...",
  "suggested_revision": "...",
  "confidence": 4
}

判断要求：
- valid：目标步骤在给定题目和上下文前提下可靠。
- invalid：目标步骤数学上不可靠，或使用了上下文没有给出的关键前提。
- uncertain：信息不足，或无法区分数学错误和表达/形式化问题。
- 如果输入包含 wrapped_claim，只能把它当作结构化证据，不要盲信。
- 如果输入包含 Lean 结果，重点看 Lean 是否暴露缺失前提、结论过强、证明失败、额外 axiom 或局部 h_missing_* 假设。
- Lean 编译失败不一定等于数学错误；需要区分形式化问题和数学问题。
- reason 必须具体说明目标步骤为什么可靠或不可靠。
- confidence 是 1 到 5。
- 所有自然语言内容使用中文。
