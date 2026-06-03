# Adaptive Adversarial Flow

本文档说明当前 adaptive adversarial 单步验证流程，以及 CoT 修改和 parser 的约束。

## 目标

对同一道数学题的某个目标 CoT 步骤进行对抗测试：

1. hacker 生成一个隐蔽但错误的 CoT 攻击版本。
2. baseline 只用自然语言判断目标步骤是否可靠。
3. lean-assisted 先用同一个自然语言判断；只有当初判为 `valid` 时，才进入 Lean 形式化和反馈判断。
4. 如果 baseline 或 lean-assisted 至少有一方没有判出 `invalid`，该 case 停止；否则继续让 hacker 基于历史反馈重试，最多 `max_rounds` 轮。

## Prompt 目录

当前流程使用：

- `prompts/adaptive_adversarial/adversarial_check_global.md`
- `prompts/adaptive_adversarial/adversarial_hacker_init.md`
- `prompts/adaptive_adversarial/adversarial_hacker_retry.md`
- `prompts/adaptive_adversarial/adversarial_judge_initial.md`
- `prompts/adaptive_adversarial/adversarial_lean_formalize.md`
- `prompts/adaptive_adversarial/adversarial_lean_pass_review.md`
- `prompts/adaptive_adversarial/adversarial_lean_fail_decide.md`
- `prompts/adaptive_adversarial/adversarial_lean_repair.md`
- `prompts/adaptive_adversarial/format_repair_json.md`
- `prompts/adaptive_adversarial/format_repair_lean.md`

## Hacker 输出

hacker 输出 JSON。核心字段：

- `attackable`: 是否适合生成攻击样本。
- `flawed_step`: 整数，表示 `modified_cot` 中哪一个 `[Step k]` 是最终要被评测的错误步骤。
- `modified_cot`: 完整的修改后 CoT。judge 后续看到这个完整 CoT。
- `why_invalid`: hacker 给出的错误理由。

CoT 只是参考，hacker 可以输出任意完整修改后 CoT，但应尽量少改。parser 会用 `flawed_step` 的编号从 `modified_cot` 中抽取目标步骤文本，作为 baseline 和 lean-assisted 共同评测的 `target_step`。

## Parser 约束

当前 parser 只负责把 hacker 输出送入评测流程，不做 CoT 语义合法性检查：

- hacker prompt 会提供完整 `original_cot` 和局部窗口。
- hacker 必须输出可解析 JSON。
- `flawed_step` 必须是正整数，`modified_cot` 和 `why_invalid` 必须是非空字符串。
- 构造 adversarial row 时，直接使用 `modified_cot` 作为 `mutated_cot`，并从对应 `[Step k]` 抽取 `target_step`。
- judge 输入优先使用完整 `mutated_cot`，而不是只用原始局部窗口。
- JSON 格式错误、hacker 字段结构错误、Lean 代码块格式错误都会回灌修复，最多 2 次。

## Thread 约定

- hacker 对同一道候选题使用同一个 Codex thread，retry 会保留上一轮攻击历史。
- baseline 和 `lean_initial` 使用同一份 initial judge prompt 和同一个用户输入模板，保证初判口径一致。
- baseline 不使用 thread，保证每轮自然语言判断是独立上下文。
- lean-assisted 对同一题同一轮使用一个 Codex thread：`lean_initial`、Lean 形式化、Lean 通过后的 review、Lean 失败后的 decide/repair 都在这个 thread 内。
- lean-assisted 不同轮使用不同 thread，避免上一轮 Lean 反馈污染下一轮判断。

## Lean-Assisted Gate

lean-assisted 的执行逻辑：

```text
lean_initial = initial judge

if lean_initial.verdict != "valid":
    返回 lean_initial 的判断，不跑 Lean
else:
    生成 Lean，或在形式化阶段直接返回 invalid
    运行 Lean check
    根据 Lean 通过/失败反馈重新判断；repair 阶段也可以直接返回 invalid
```

因此，只有模型初步认为目标步骤正确时，Lean 才作为二次审查介入。
但进入 Lean 流程后，每个子步骤的主要目标仍然是检查目标步骤是否错误；formalize、fail_decide、pass_review、repair 都可以在发现目标步骤不可靠时直接返回 `invalid` 和理由。

## 结果状态

- `too_obvious`: baseline 和 lean-assisted 都判 `invalid`，攻击太明显，继续对抗。
- `lean_rescue`: baseline 没判出 `invalid`，lean-assisted 判出 `invalid`，这是目标样本。
- `lean_missed`: baseline 和 lean-assisted 都没判出 `invalid`。
- `lean_weaker_than_baseline`: baseline 判出 `invalid`，lean-assisted 没判出。
- `model_rescue_no_lean`: lean-assisted 没跑 Lean，但初判判出 `invalid`。

## 已发现的旧结果问题

旧 parser 曾只把 `context_steps` 窗口交给 judge。若 hacker 修改窗口外 step，该修改不会出现在 judge 输入里。这样得到的 “baseline 和 lean-assisted 都被骗过” 结果不能作为有效样本。

修复后，hacker 直接输出完整 `modified_cot`，judge 会看到完整 `mutated_cot`，不会再因为窗口截断静默丢掉上下文修改。
