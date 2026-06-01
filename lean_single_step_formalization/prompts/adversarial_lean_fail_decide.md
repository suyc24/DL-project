Lean 没有通过。你要决定：这是否已经暴露目标步骤错误，还是应该继续修复 Lean 代码。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "action": "return_invalid | continue_repair | return_uncertain",
  "verdict": "invalid | uncertain",
  "issue_type": "missing_premise | too_strong | algebra_error | inequality_direction | quantifier_swap | modular_condition | boundary_case | formalization_issue | other",
  "reason": "...",
  "repair_instruction": "...",
  "confidence": 4
}

判断标准：
- 如果 Lean 错误直接指向目标命题不可证、需要明显错误的系数等式、缺失关键前提、边界条件不成立，应 action=return_invalid。
- 如果 Lean 错误主要是语法、类型、库接口、定理名、tactic 写法问题，应 action=continue_repair。
- 如果无法区分数学错误和形式化问题，应 action=return_uncertain。
- 不要因为 Lean 没过就自动判 invalid。
- 也不要因为可以通过添加强 h_missing_* 就继续修；如果 h_missing_* 正是目标步骤遗漏的关键数学条件，应返回 invalid。
- repair_instruction 只有在 action=continue_repair 时写具体修复方向；否则写空字符串。
- 所有自然语言内容使用中文。
