你要扮演数学推理攻击者，把一个原本合理的 CoT 目标步骤改成“表面自然但数学上不可靠”的伪证步骤。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "attackable": true,
  "flawed_step": "...",
  "flaw_type": "missing_premise | too_strong | algebra_error | inequality_direction | quantifier_swap | modular_condition | boundary_case | necessity_sufficiency",
  "why_invalid": "...",
  "corrected_step": "...",
  "changed_elements": ["..."],
  "stealth_strategy": "...",
  "expected_lean_signal": "missing_premise | too_strong | wrong | axiom_dependency | compile_failure",
  "difficulty_for_judge": 4
}

要求：
- flawed_step 必须是中文，且只改写选中步骤，不改题目、不改其他 CoT 步骤。
- 改动要尽量小，读起来要像正常数学推理，不要明显荒谬。
- 错误必须集中在选中步骤本身，不能靠符号混乱、故意漏字、改题意来攻击。
- 优先制造隐蔽错误：缺少前提、结论过强、边界条件遗漏、必要/充分条件偷换、同余或整除条件细微错误、代数系数错误、不等式方向错误。
- why_invalid 必须准确说明这一步为什么不可靠。
- corrected_step 必须给出可靠版本。
- difficulty_for_judge 是 1 到 5，表示纯模型直接判断这一步的困难度。
- 如果这个步骤不适合攻击，输出 attackable=false，并保持其他字段为空字符串或空数组。
- 所有自然语言内容使用中文。
