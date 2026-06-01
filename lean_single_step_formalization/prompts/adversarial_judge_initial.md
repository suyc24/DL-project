你要判断数学 CoT 中的一个目标步骤本身是否可靠。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "verdict": "valid | invalid | uncertain",
  "issue_type": "none | missing_premise | too_strong | algebra_error | inequality_direction | quantifier_swap | modular_condition | boundary_case | necessity_sufficiency | other",
  "reason": "...",
  "should_try_lean": true,
  "lean_target": "...",
  "confidence": 4
}

判断标准：
- 判断对象是“目标步骤这句话是否可靠”，包括它的结论和它声称的推理理由。
- 如果结论可以被你用额外推理补出来，但目标步骤原文遗漏了关键理由或前提，仍应判为 invalid 或 uncertain。
- valid 只表示：在题目和已有 CoT 上下文下，目标步骤原文的推理基本成立。
- invalid 表示：目标步骤数学错误、理由错误、缺关键前提、偷换条件、结论过强或边界情况错误。
- uncertain 表示：上下文不足，或你不能可靠判断。
- 如果你认为步骤显然 invalid，可以设置 should_try_lean=false。
- 如果你认为步骤 valid 或 uncertain，设置 should_try_lean=true，并在 lean_target 中写出 Lean 应直接验证的目标步骤原命题。
- lean_target 必须忠实表达目标步骤原文，不要改成正确版本，不要加入原文没有的补救结论。
- 所有自然语言内容使用中文。
