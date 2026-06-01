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
- 你的任务是“包装目标步骤声称的局部命题”，不是判断它真假。
- conclusion 必须表达选中步骤原文声称要推出的结论；不要把 conclusion 改成“该步骤不成立”“一般不成立”“正确版本是...”。
- 不要在 wrapped_claim 中纠错、改写成否定命题、给反例、或替换为正确结论。
- proof_description 只能描述原 CoT 试图如何证明该 conclusion；即使你怀疑该步有错，也不要在 proof_description 中诊断错误或给正确版本。
- 不能把 conclusion 直接作为 premise。
- 如果选中步骤只是记号、改名、转场、重复结论，设置 low_value=true；但仍尽量结合周围 CoT 包装成有意义的 premise -> conclusion。
- 如果题目里的表达式可能为负，wrapped_claim 必须明确使用整数语义，避免自然数截断减法。
- 所有自然语言内容使用中文。
