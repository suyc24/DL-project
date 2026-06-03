你要修复上一轮模型输出的格式错误。

只返回以下两种结果之一：

1. Lean 代码块，包含完整文件和 `import Mathlib`。
2. invalid JSON：
{
  "verdict": "invalid",
  "reason": "...",
  "confidence": 4
}

要求：
- 保留原始任务要形式化的数学语义，不要改成自然语言诊断。
- 如果上一轮实际是在指出目标步骤错误，把它修成合法 invalid JSON，不要强行改成 Lean。
- 使用指定 theorem 名字。
- 不使用 `sorry`、`admit`。
- 不要输出 Markdown 说明或额外解释。
