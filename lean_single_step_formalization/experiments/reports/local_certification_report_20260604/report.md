# Lean 局部证明认证实验报告

日期：2026-06-04
对象：`adaptive_adversarial_gv_v2` 局部 Lean formalization / generator-verifier pipeline

## 1. 核心结论

我们现在能合理主张的贡献不是“自动验证整道题的自然语言证明”，而是：

> 在数学家或上游系统给定局部 proof step 的前提、推理动作和目标结论后，系统把它转成严格忠实的 Lean 局部证明义务，并用 compile + verifier review 判断该局部义务是否成立。

这对数学家有用的地方是：它把一句自然语言 proof step 变成一个可审计的 proof obligation。若 Lean 编译通过且 verifier 判定忠实，则该局部步骤在给定前提下得到 machine-checkable certification；若失败，系统会给出缺失前提、错误等式、分支覆盖不足、量词/构造不匹配等局部原因。

边界也必须写清楚：系统不保证 `step_decomposition` 本身完整覆盖原题，也不保证用户给出的 premises 足以服务整篇证明。全局条件是否合适，应由数学家或另一个 global-bridge checker 判断。本报告中的实验支持的是“局部认证”这个 claim。

## 2. 当前 v2 pipeline

输入仍是 `target_step` 和 `step_decomposition`。最新版 `step_decomposition` schema 为：

```json
{
  "premises": [{"id": "P1", "text": "...", "source": "problem | previous_context | target_step"}],
  "proof_steps": [{"id": "S1", "text": "...", "uses": ["P1"], "yields": "..."}],
  "conclusion": "...",
  "confidence": 4
}
```

重要修改：删除 `implicit_dependency`。所有写入 `premises` 的条件都必须被视为已经由题目、前文或目标步骤开头明确给出；如果缺少前提，不能补进 premises，而应让缺口保留在 `proof_steps` 与 `conclusion` 中。

Generator 规则：

- 只在一个持续 thread 中 formalize/repair，保留前后轮上下文。
- 必须完全忠实于 `target_step` 和 `step_decomposition`。
- 在严格忠实前提下尽量修到 Lean compile ok。
- 只有发现原步骤本身在数学上不成立，才返回 `compile_ok=false`，并说明具体语义缺口。
- 不能为了编译新增关键前提、替换构造、改变量词、把目标结论当前提、使用 `sorry/admit/axiom` 掩盖证明义务。

Verifier review 规则分成两个 prompt 文件：

- `verifier_review_compile_ok.md`：generator 报告 `compile_ok=true` 时，只检查 Lean 是否严格忠实；忠实则 `valid`，不忠实则 request repair 回灌给 generator。
- `verifier_review_compile_fail.md`：generator 报告 `compile_ok=false` 时，判断失败理由是否足以说明自然语言局部步骤 invalid；理由合理则 `invalid`，否则 request repair。

线程规则：generator 永远是一个持续 thread；每一次 verifier review 都使用新 thread，避免 verifier 的历史上下文污染下一次审查。

## 3. 实验概览

| 实验 | run | 目的 | 模型/并行 |
|---|---|---|---|
| OPC false-step v2 fair run | `codex55_high_opc_stepd_gv_v2_50cases_001` | 与 baseline 比较，在已知 OPC invalid annotation 上看 GV 是否抓错 | `gpt-5.5`, high, 5 并行 |
| OPC positive-control 构造 | `opc_positive_stepd_controls_codex55_high_50_002` | 从 OPC 中人工/模型辅助筛出 50 个正确局部步骤 | `gpt-5.5`, high |
| Positive-control no-baseline GV | `positive_gv_v2_no_baseline_codex55_high_50_001` | 测试正确局部 step 是否被判为 valid | `gpt-5.5`, high, 5 并行 |
| Missing-premise use case | `missing_premise_usecase_gv_v2_codex55_high_10groups_001` | 模拟用户少给 0/1/2/3 个前提时 GV 的反应 | `gpt-5.5`, high, 10 并行 |
| Structured hacker pre-fix | `structured_stepd_hacker_gv_v2_codex55_high_attached_case{0,1,2}_001` | 测试旧 schema 下隐蔽伪证攻击 | `gpt-5.5`, high |
| Structured hacker noimplicit | `structured_stepd_hacker_gv_v2_noimplicit_*` | 删除 `implicit_dependency` 后测试 hacker | `gpt-5.5`, high, max 10 rounds |
| DeepSeek partial | `deepseek_v4pro_thinking_opc_stepd_gv_v2_50cases_001` | 模型替换对照，已中止，仅作参考 | DeepSeek v4 pro thinking |

## 4. OPC false-step 结果

主 run：`runs/codex55_high_opc_stepd_gv_v2_50cases_001`

| 指标 | 数值 |
|---|---:|
| cases | 50 |
| baseline invalid | 48 |
| GV invalid | 47 |
| GV valid | 3 |
| baseline annotation alignment avg | 8.38 |
| GV annotation alignment avg | 8.04 |
| weak cases | 5 |

这些 false-step case 来自 OPC 的 invalid annotation。表面看 baseline annotation alignment 略高，GV 有 3 个 valid。但人工复核后发现，这些 GV-valid 里有一类不是 Lean 放过错误，而是 `step_decomposition` 构造出的局部命题本身是对的。例如之前讨论过的 `mth power`：原题或原证明层面可能存在 global mismatch，但 step decomposition 暴露给 Lean 的局部命题可以成立，因此局部 GV 判 valid 是合理的。

因此，这个实验说明两点：

- 如果目标是“复现 OPC annotation 的原始错误诊断”，baseline 仍然很强。
- 如果目标是“检查给定局部 proof obligation 是否成立”，不能把所有 annotation mismatch 都算作 GV 错；需要区分 local validity 和 global faithfulness。

这个实验推动了后续结论：Lean/GV 应定位为局部认证器，而不是整题自然语言审判器。

## 5. Positive-control 结果

构造 run：`runs/opc_positive_stepd_controls_codex55_high_50_002`

| 指标 | 数值 |
|---|---:|
| candidate pool | 220 |
| processed | 66 |
| accepted | 50 |
| rejected by annotation/local audit | 16 |
| stepD errors | 0 |

GV run：`runs/positive_gv_v2_no_baseline_codex55_high_50_001`

| 指标 | 数值 |
|---|---:|
| cases | 50 |
| GV valid | 50 |
| GV invalid | 0 |
| false invalid | 0 |
| final compile ok | 50 |
| avg generator attempts | 1.3 |
| avg verifier reviews | 1.3 |

这是最重要的正例控制实验。它说明：当 `target_step` 和 `step_decomposition` 是人工筛过的正确局部 proof obligation 时，v2 GV 不会过度怀疑，50/50 全部返回 valid 且 Lean 最终 compile ok。

这支持一个关键 claim：GV 的 invalid 判定不是简单的“形式化困难”或“Lean 不会证明就判错”。在正确局部步骤上，系统能稳定地产生忠实 Lean certification。

## 6. Missing-premise use case

run：`runs/missing_premise_usecase_gv_v2_codex55_high_10groups_001`

该实验模拟真实用户使用场景：用户想验证一个局部 step，但少给了一部分 premises。我们构造 10 组正确局部链式证明，每组分别移除 0/1/2/3 个 premise，共 40 cases。

| missing premises | cases | GV verdict | compile status |
|---:|---:|---|---|
| 0 | 10 | 10 valid | 10 compile ok |
| 1 | 10 | 10 invalid | 10 compile fail with semantic reason |
| 2 | 10 | 10 invalid | 10 compile fail with semantic reason |
| 3 | 10 | 10 invalid | 10 compile fail with semantic reason |

典型例子：integer additive chain。

- 完整输入：`u=1`, `v=u+1`, `w=v+2`，目标 `w=4`。GV 返回 valid，Lean 用三个等式推出结论。
- 少 1 个 premise：只给 `u=1`, `v=u+1`，目标仍是 `w=4`。GV 返回 invalid；理由是 `w` 没有任何约束，取 `u=1, v=2, w=0` 时前提成立但结论不成立。

这个实验直接支持产品层面的 use case：如果数学家少给前提，系统不会默认补前提，也不会把缺前提藏进 Lean theorem assumption；它会返回 invalid，并指出局部证明义务中哪一环无法推出。

## 7. Structured hacker 结果

Hacker 模式目标：从正确 positive-control case 出发，构造结构仍像 `step_decomposition` 的隐蔽伪证。若 GV 判 invalid，则把 invalid reason 回馈给 hacker，让它再攻击，最多 10 轮。

### 7.1 pre-fix 教训

旧 schema 允许 `implicit_dependency`，这导致一次真实系统 miss：

| case | raw status | 人工判读 |
|---|---|---|
| attached case 0 | GV valid after 5 rounds | 真实 miss：hacker 把缺失条件放进 `implicit_dependency`，Lean 当成可用前提 |
| attached case 1 | detected invalid after 10 rounds | GV 抓住 |
| attached case 2 | GV valid after 1 round | false attack：局部步骤本身成立 |

结论：`implicit_dependency` 是危险入口。只要 schema 允许“默认依赖”，hacker 就可以把真正缺失的前提伪装成默认已知条件。后续 prompt 删除该字段，并要求 premises 只能来自明确给定来源。

### 7.2 noimplicit 后的 13 个 hacker case

相关 runs：

- `structured_stepd_hacker_gv_v2_noimplicit_codex55_high_case{0,1,2}_001`
- `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset{3..12}_001`

raw status 汇总：

| raw status | cases |
|---|---:|
| `hack_success_gv_valid` | 8 |
| `hack_detected_invalid` | 4 |
| `hacker_gave_up` | 1 |

但 raw status 不能直接当作系统成败。人工审计 8 个 `hack_success_gv_valid` 后，全部是 false attack / benign local step，没有发现合理伪证被 GV 放过。

重点人工判读：

| case | raw status | 人工判读 |
|---|---|---|
| noimplicit case 1 | GV valid | false attack：局部反射构造存在，hacker 抱怨的是下游接口 |
| noimplicit case 2 | GV valid | false attack：witness/coverage 在局部命题中成立 |
| offset 3 | GV valid | false attack：选择稀疏 prime injection 是有效存在性构造 |
| offset 4 | GV valid | false attack：`s=1` 的 normalized sum 由上下文支持 |
| offset 5 | detected invalid | GV 抓住奇合数枚举遗漏 25 |
| offset 6 | hacker gave up | 未形成有效攻击 |
| offset 8 | detected invalid | GV 抓住 boundary/global maximum 缺口 |
| offset 9 | detected invalid | GV 抓住 Bezout/gcd 推出过强非负有界解 |
| offset 10 | GV valid | false attack：hacker 自己的 `why_invalid` 实际承认无 flaw |
| offset 11 | GV valid | false attack：partite hypergraph matching 已包含 coordinatewise disjointness |
| offset 12 | GV valid | false attack：目标使用剩余 admissible primes infinite/unbounded，局部忠实 |

结论：删除 `implicit_dependency` 后，在目前 13 个 noimplicit hacker case 中，没有确认的真实系统 miss。这个结果不是形式化安全证明，但说明 prompt/schema 修补有效，hacker 很难再通过“偷偷补前提”攻击局部 Lean certification。

## 8. DeepSeek partial run

run：`runs/deepseek_v4pro_thinking_opc_stepd_gv_v2_50cases_001`

该 run 已中止，只完成 30 cases：

| 指标 | 数值 |
|---|---:|
| cases | 30 |
| baseline invalid | 28 |
| GV invalid | 26 |
| GV valid | 4 |
| baseline alignment avg | 8.20 |
| GV alignment avg | 7.17 |
| weak cases | 8 |

此结果不作为主结论，只说明模型替换后稳定性下降，需要单独调 prompt 和成本预算。当前 report 的主证据仍使用 `gpt-5.5 + high`。

## 9. 通信文件和输入输出

普通 GV case 主要文件：

- `step_decompose/step_decompose.prompt.md`：stepD 输入 prompt。
- `step_decompose/step_decompose.response.txt`：stepD 原始响应。
- `step_decompose/step_decomposition.json`：结构化局部 proof obligation。
- `lean_gv_v2/generator_formalize.prompt.md`：generator formalization prompt。
- `lean_gv_v2/generator_formalize.response.txt`：generator 原始响应。
- `lean_gv_v2/generator_report_*.json`：generator 结构化报告，含 `compile_ok`、reason、Lean 代码/文件信息。
- `lean_gv_v2/verifier_review_*_compile_ok.prompt.md` 或 `*_compile_fail.prompt.md`：按 compile 分支选择的 verifier prompt。
- `lean_gv_v2/verifier_review_*.response.txt`：verifier 原始响应。
- `lean_gv_v2/lean_assisted_result.json`：GV 最终结果。
- `lean_gv_v2/outputs.md`：面向人工阅读的汇总输出。
- `threads/generator.thread`：generator 持续 thread。
- `threads/verifier_reviews/*.thread`：每次 verifier review 的新 thread。

Hacker case 主要文件：

- run root `input/base_positive_rows.jsonl`：hacker 的正例来源。
- run root `input/gv_invalid_seed_examples.json`：从 GV invalid 中抽的攻击启发例。
- per case `base_row.json`：本 case 的原始正确局部 step。
- per case `hacker.thread`：hacker 自己的持续 thread。
- per round `hacker.prompt.md` / `hacker.response.txt`：本轮攻击输入与原始输出。
- per round `hacker_attack.json`：hacker 结构化攻击，含 `target_step`、`step_decomposition`、`flaw_type`、`why_invalid`、`stealth_strategy`。
- per round `adversarial_row.json`：送入 GV 的攻击样本。
- per round `adversarial_step_decomposition.json`：送入 GV 的结构化 stepD。
- per round `lean_gv_v2/*`：该攻击样本的 GV 运行产物。
- per round `round_result.json`：本轮 GV 结果和是否继续攻击。
- per case `case_result.json`：hacker 最终状态。
- run root `summary.json`：run 级汇总。

缺 premise use case 主要文件：

- run root `input_rows.jsonl`：40 个输入样本。
- per case `row.json`：该 case 的 premises、被移除 premise、目标结论。
- per case `case_result.json`：GV verdict、reason、Lean evidence、compile status。
- per case `lean_gv_v2/*`：generator/verifier 全部通信文件。

## 10. 人工审计明细

详细 case-level 人工判断见 `manual_audit_details.md`。这里列最重要结论：

| 类别 | 人工审计结论 |
|---|---|
| positive-control false-invalid | 50 个正确局部 step 中 false-invalid 为 0；没有 case id。 |
| OPC weak/GV-valid | 5 个审计 case 中没有 true system error；主要是 metric mismatch、local/global mismatch 或 external theorem applicability。 |
| missing-premise | missing=0 全 valid；missing=1/2/3 全 invalid，人工判断为正确拒绝，不是 false-invalid。 |
| hacker pre-fix | `implicit_dependency` 导致 1 个 confirmed miss，已通过删除该字段修复。 |
| hacker noimplicit | raw `hack_success_gv_valid` 的 8 个 case 经人工审计均为 false attack / benign local step；4 个 detected invalid 是 true detection；1 个 gave up。 |

最容易误读成 false-invalid 的是 OPC 中两个低 alignment 的 invalid case：

- `003_3_OPC_best_of_n_BMOSL_2018_12_13`：GV invalid 是合理的，因为它抓到 inequality-transfer 错误；只是 annotation reason 不同。
- `010_4_OPC_best_of_n_BMOSL_2017_22_49`：GV invalid 是合理的，因为它给出 convexity maximization 反例；annotation 只说 computational steps skipped。

因此当前报告口径下，confirmed false-invalid = 0。GV valid on annotation-invalid 也不能算 false-invalid；那类应标为 local-valid/global-bridge mismatch。

## 11. 应该如何向别人说明 Lean 有用

推荐表述：

> 我们不声称 Lean 自动理解或验证整篇自然语言证明。我们做的是局部 proof-step certification：给定明确的局部前提和目标，系统生成严格忠实的 Lean obligation，并用 Lean 编译与 verifier review 检查该局部步骤是否真的由这些前提推出。这个过程能发现缺 premise、偷换结论、分支覆盖不足和构造不忠实，也能在正确局部步骤上给出 machine-checkable certificate。

这比单纯 LLM judge 有两个优势：

- 它有可执行证明对象：valid 不只是自然语言判断，而是有 Lean obligation 和 proof。
- 它能暴露前提缺口：少给 premise 时，系统返回具体反例或无法推出的语义点，而不是自动补全。

## 12. 限制

当前系统的限制：

- `step_decomposition` 的全局正确性不在本 report 主 claim 内。若 stepD 抽错目标、漏掉上下文或把局部目标构造得太弱，Lean 只会认证这个被给定的局部义务。
- 数学家仍需判断 premises 是否确实来自原题/前文，conclusion 是否正是需要证明的下一步。
- Lean proof 可以证明局部 theorem，但不能自动说明这个 theorem 在整题证明结构中的位置正确。
- 对复杂定义，目前仍可能用局部 predicate/structure 表示接口；这需要继续审计是否过粗。
- Hacker 实验需要人工判读 raw `GV valid`，因为 hacker 可能提出并不真正 invalid 的攻击。
- Missing-premise 实验是合成 use case，覆盖了前提敏感性，但还不是全部真实 olympiad proof 风格。

## 13. 下一步最重要实验

优先级最高的是“真实 OPC positive-control 的 premise ablation”：

1. 从 50 个 accepted positive-control OPC 局部步骤中抽样。
2. 人工确定每个局部 theorem 的必要 premises。
3. 系统性删除 1 个关键 premise、删除多个 premise、删除非关键 premise。
4. 观察 GV 是否对关键 premise 缺失返回 invalid，对非关键 premise 缺失仍能 valid。

这个实验比当前 synthetic missing-premise 更接近真实用户场景，也能直接支撑“给定条件不完整时，系统会指出缺口”。

第二优先级是 global-bridge checker：

- 输入原题、前文、`target_step`、`step_decomposition`。
- 不跑 Lean，专门判断 stepD 的 premises 是否确实来自上下文、conclusion 是否是原 proof 需要的局部目标。
- 与当前 Lean local certification 组合后，才能更接近全 pipeline claim。

第三优先级是扩大 noimplicit hacker：

- 至少 50 个 hacker case。
- 每个 raw `GV valid` 必须人工标注：confirmed miss / false attack / ambiguous。
- 报告 confirmed miss rate，而不是 raw hack_success rate。

## 14. artifact 索引

本报告包目录：

`/root/DL-project/lean_single_step_formalization/experiments/reports/local_certification_report_20260604`

子目录：

- `runs/`：本报告引用的 run symlink。
- `docs/`：旧报告、人工 label 和参考文档 symlink。
- `prompts/`：当前 `adaptive_adversarial_gv_v2` prompt symlink。
- `scripts/`：本轮新增/使用的 runner script symlink。
- `manifest.md`：逐项 artifact 列表。
