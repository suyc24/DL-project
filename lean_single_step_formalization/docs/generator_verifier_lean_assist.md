# Generator/Verifier Lean-Assisted Runner

新入口：

```bash
python3 scripts/run_adversarial_game_gv.py \
  --config configs/default.json \
  --candidates <candidates.jsonl> \
  --run-id <run_id>
```

Prompt 目录：

```text
prompts/adaptive_adversarial_gv/
```

## Thread 设计

这个 runner 的 lean-assisted 内部只有两个角色 thread：

- `generator.thread`: 负责生成 Lean、修复 Lean；如果在写 Lean 或修 Lean 时发现目标步骤不可靠，可以直接返回 `invalid`。
- `verifier.thread`: 负责 initial 判断、检查 generator 的 Lean 代码和 Lean 输出、决定 `valid` / `invalid` / `request_repair`；它也可以随时返回 `invalid`。

hacker 仍复用现有 adversarial hacker 流程和 candidate-level hacker thread。baseline 不在 lean-assisted 内部，不额外占用 generator/verifier thread。

generator 和 verifier 都会拿到一个真实工作区。generator 直接在工作区里写 `.lean` 文件并运行 `lake env lean <file>`；verifier 可以查看该文件，也可以自己重新运行编译命令。generator 不需要在回复中输出 Lean 代码，只返回 JSON 报告。

## Lean-Assisted 流程

```text
verifier initial
  if invalid: stop
  if valid: ask generator to formalize

generator formalize
  if return_invalid JSON: stop
  if generated JSON: verifier 检查工作区中的 Lean 文件和编译报告

verifier review
  if return_valid: stop valid
  if return_invalid: stop invalid
  if request_repair: ask generator to repair

generator repair
  if return_invalid JSON: stop invalid
  if generated JSON: return to verifier review
```

所有自然语言判断和 prompt 内容都使用中文。
