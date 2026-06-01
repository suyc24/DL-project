你要把选中的 CoT 步骤包装成一个局部命题，供下一步 Lean 形式化使用。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "low_value": false,
  "low_value_reason": "",
  "used_cot": [
    {
      "step_id": 1,
      "is_selected": true,
      "text": "...",
      "role": "premise | conclusion | context"
    }
  ],
  "wrapped_claim": {
    "premises": [
      {
        "text": "...",
        "source": "cot_step_N | problem | standard_math"
      }
    ],
    "conclusion": {
      "text": "...",
      "source": "cot_step_N | problem | standard_math"
    },
    "proof_description": "详细说明如何从前提推出结论。必须解释每个前提如何使用，不能只写显然。"
  }
}

硬性要求：
- used_cot 必须包含选中的 CoT 步骤，且恰好一个 is_selected 为 true。
- 选中的 CoT 步骤 text 必须原样出现在 used_cot 里。
- premises 的每一项都必须有 source，source 只能来自 cot_step_N、problem、standard_math。
- conclusion 必须覆盖选中步骤的数学含义。
- 不能把 conclusion 直接作为 premise。
- 如果选中步骤只是记号、改名、转场、重复结论，设置 low_value=true；但仍尽量结合周围 CoT 包装成有意义的 premise -> conclusion。
- 如果题目里的表达式可能为负，wrapped_claim 必须明确使用整数语义，避免自然数截断减法。
- 所有自然语言内容使用中文。
