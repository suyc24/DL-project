你要给一条数学 CoT 的每一步打分，用于选择哪些步骤最值得做单步 Lean 验证。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "steps": [
    {
      "step_id": 1,
      "is_mathematical_claim": true,
      "low_value": false,
      "verification_value": 4,
      "risk": 3,
      "lean_feasibility": 4,
      "reason": "..."
    }
  ]
}

打分含义：
- is_mathematical_claim：该步是否包含明确数学断言。
- low_value：是否只是定义、重述题目、转场、计划、尝试方向、显然小算术。
- verification_value：1 到 5，Lean 验证这一步对发现推理问题的价值。
- risk：1 到 5，这一步发生幻觉、跳步、条件缺失或结论过强的风险。
- lean_feasibility：1 到 5，局部包装成 premise -> conclusion 后用 Lean 验证的可行性。

选择偏好：
- 高价值：整除、同余、不等式、极值、构造正确性、归纳步骤、组合计数转化、代数恒等式、关键分类讨论。
- 低价值：只引入记号、只重复题目、只说“我们需要证明”、只给最终答案、只做非常简单的单调推论。
- 高层概念可以高分，只要能用局部谓词或接口形式化。

硬性要求：
- 必须给输入中的每一个 step_id 打分。
- step_id 必须与输入一致。
- 所有分数必须是 1 到 5 的整数。
- reason 使用中文，简短说明。
