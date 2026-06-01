你要根据 Lean 单步验证结果，判断原 CoT 中被选中的这一步是否可靠，以及 Lean 反馈是否指出了可修正的问题。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "step_status": "valid | missing_premise | too_strong | wrong | low_value | formalization_issue",
  "issue_found": true,
  "diagnosis": "...",
  "revised_step": "...",
  "added_premises": ["..."],
  "impact_on_solution": "...",
  "revised_final_answer": "..."
}

判断标准：
- valid：原步骤和 wrapped_claim 都基本可靠，Lean 没暴露实质问题。
- missing_premise：原步骤可能正确，但需要补充前提、引理、边界条件或定义接口。
- too_strong：wrapped_claim 或原步骤的结论说得比 CoT/题目实际能支持的更强。
- wrong：原步骤数学上有错误。
- low_value：该步骤只是定义、重述、计划、转场或过小的显然推论，不值得 Lean 验证。
- formalization_issue：Lean 错误主要来自形式化表达、mathlib 接口或语法问题，不足以说明原数学步骤错。

要求：
- diagnosis 必须具体说明 Lean 反馈如何支持你的判断。
- revised_step 必须用中文重写选中步骤；如果原步骤无需修改，可原样保留并说明。
- added_premises 列出为了使该步可靠需要补充的前提；没有则为空数组。
- impact_on_solution 说明该问题是否影响后续推理和最终答案。
- revised_final_answer 如果不需要改最终答案，写空字符串。
- 不要盲目相信 Lean 失败等于数学错误；要区分数学问题和形式化问题。
- 如果 Lean 使用了 `h_missing_*` 或 `obligation_*`，要解释这些依赖是不是合理的局部接口，还是暴露了缺前提/过强结论。
