你要修复上一轮 generator 输出的 JSON 格式错误。

只返回一个合法 JSON 对象，不要输出 Markdown，不要输出代码块，不要解释。

generator 报告格式必须是：
{
  "action": "generated",
  "lean_file": "...",
  "compile_command": "...",
  "compile_ok": true,
  "stdout_tail": "...",
  "stderr_tail": "...",
  "faithfulness_summary": "...",
  "reason": "...",
  "confidence": 4,
  "lean_code": "..."
}

要求：
- `action` 必须是 `"generated"`；禁止输出 `return_invalid`。
- `compile_ok` 必须是 boolean，真实表达 Lean 是否编译通过。
- 保留原始语义，不要重新发明 Lean 代码或数学判断。
- 如果原始输出是在说明自然语言步骤有明确问题，则保留为 `generated` 且 `compile_ok=false`，把理由放进 `reason` 和/或 `stderr_tail`。
- 如果原始输出包含完整 Lean 文件内容，放入 `lean_code` 字符串字段。
- 不要输出 Lean 代码块；若需要 Lean 内容，只能作为 JSON 字符串值。
