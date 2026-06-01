你要修复一份没有通过编译的 Lean 4 代码。

只返回一个 Lean 代码块，包含完整文件和 `import Mathlib`。

输入会包含：
- 原 Lean 代码；
- Lean 输出/错误；
- fail_decide 阶段给出的 repair_instruction。

要求：
- 保持 theorem 名字不变。
- 保持目标步骤原命题不变，不要把目标改成正确版本、反例命题、或“该步骤错误”。
- 不使用 `sorry`、`admit`。
- warning 不需要处理，只修 error。
- 如果只是语法、类型、tactic、Nat/Int/Real 转换、库接口问题，请直接修复。
- 如果需要补充前提，只能加入具体的局部 `h_missing_*` 假设。
- `h_missing_*` 不能直接断言最终结论，不能制造矛盾后用 `False.elim` 证明目标。
- 不要加入与已有前提不相容的 `h_missing_*`。
- 不要用全局 axiom 直接掩盖目标步骤；只有高层标准接口无法表达时，才允许窄范围 `obligation_*` axiom。
- 如果你发现目标命题实际上不可证明，也不要在本阶段输出自然语言诊断；仍然返回最忠实的 Lean 尝试，让下一轮 fail_decide 判断。
- 所有注释使用中文。
