你的主要目标仍然是检查目标步骤是否错误。

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 格式必须是：
{
  "action": "return_valid | return_invalid | request_repair",
  "verdict": "valid | invalid",
  "reason": "...",
  "lean_evidence": "...",
  "repair_instruction": "...",
  "confidence": 4
}

Lean 处理：
- 逐字检查自然语言版本证明中目标步骤，在lean中是否有忠实的对应，如果没有，考虑正确性判定。请逐字查看，严格对照。
- 逐字检查lean语言中的每一个字段在自然语言中是否有忠实的设定，还是lean为了过编译而添加的内容。

正确性判定：
- 当你发现有语义不同的内容时，判断这是自然语言版本的错误或者内容缺失，还是lean版本没有忠实反应正确的自然语言版本。
