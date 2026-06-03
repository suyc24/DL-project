Lean 没有通过。你的主要目标仍然是判断目标步骤是否错误；只有当问题明显是 Lean 表达或代码问题时，才继续修复 Lean 代码。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "action": "return_invalid | continue_repair",
  "verdict": "invalid | valid ",
  "reason": "...",
  "repair_instruction": "...",
  "confidence": 4
}

判断标准：
- 如果你依然觉得这个自然语言的目标步骤陈述是对的，只是lean没有写好，使用continue_repair
- 如果你发现了自然语言的目标步骤中的错误，使用return_invalid
