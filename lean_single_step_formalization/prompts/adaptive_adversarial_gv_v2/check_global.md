全局目标：用 Lean 辅助判断数学 CoT 的目标步骤是否严格可靠。

核心协议：
- 输入中的 step_decomposition 是中性结构化材料，不是正确性证明。
- generator 必须在严格语义忠实的前提下尽量产出可编译 Lean；不能为了编译通过而修正、加强、弱化、替换或重写自然语言步骤。
- generator 只返回 `action="generated"`，用 `compile_ok` 表示 Lean 是否真实编译通过。
- generator 只有在发现目标步骤原文存在明确数学问题、使忠实 Lean 无法成立时，才应返回 `compile_ok=false`，并在 `reason` 中说明具体问题。
- `compile_ok=false` 不是最终 verdict；review 需要独立判断 generator 的失败理由是否合理。
- review 分两类：compile-ok review 只审查 Lean 是否严格忠实；compile-fail review 审查失败理由是否足以说明自然语言步骤 invalid。
- 判断对象始终是目标步骤原文，不是一个补全后、更强或更漂亮的新证明。
- 所有自然语言字段使用中文。
