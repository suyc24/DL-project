你要修复上一轮 generator 输出的 JSON 格式错误。

只返回一个合法 JSON 对象，不要输出 Markdown，不要输出代码块，不要解释。

如果 generator 已写入 Lean 文件，格式是：
{
  "action": "generated",
  "lean_file": "...",
  "compile_command": "...",
  "compile_ok": true,
  "stdout_tail": "...",
  "stderr_tail": "...",
  "faithfulness_summary": "...",
  "reason": "...",
  "confidence": 4
}

如果 generator 发现目标步骤错误，格式是：
{
  "action": "return_invalid",
  "verdict": "invalid",
  "reason": "...",
  "confidence": 4
}

要求：
- 只修 JSON 格式，不重新发明答案。
- 保留原始语义。
- generated JSON 必须包含 `faithfulness_summary`。
- `faithfulness_summary` 要说明 Lean 是否逐项对应目标步骤原文。
- 如果原始输出实际是在指出目标步骤错误，修成 return_invalid。
- 不要输出 Lean 代码。
