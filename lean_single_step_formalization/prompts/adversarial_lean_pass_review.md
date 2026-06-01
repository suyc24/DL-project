Lean 编译通过了。你要根据 Lean 代码和 Lean 依赖信息，重新判断目标步骤是否可靠。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "verdict": "valid | invalid | uncertain",
  "issue_type": "none | missing_premise | too_strong | formalization_gap | other",
  "reason": "...",
  "lean_evidence": "...",
  "confidence": 4
}

判断标准：
- Lean 通过不等于目标步骤一定正确。
- 你必须检查 Lean theorem 是否忠实表达目标步骤原文。
- 如果 Lean 证明依赖 `h_missing_*`，判断这些前提是否已经在题目或 CoT 上下文中明确给出。
- 如果 `h_missing_*` 是为了补足目标步骤遗漏的关键条件，应判为 invalid 或 uncertain，而不是 valid。
- 如果使用了 `obligation_*` axiom，判断它是合理标准接口，还是掩盖了目标步骤的关键证明。
- 如果 Lean theorem 被弱化、改成正确版本、改成反例、或没有覆盖目标步骤原文，应判为 uncertain 或 invalid，并说明是 formalization_gap。
- 只有当 Lean 目标忠实、依赖合理、且没有暴露缺失关键前提时，才判 valid。
- 所有自然语言内容使用中文。
