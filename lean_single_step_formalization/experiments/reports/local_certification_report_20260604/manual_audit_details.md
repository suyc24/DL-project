# Manual Audit Details

本文件列出报告中用到的人工审计判断，重点说明哪些 case 不是 `false invalid`。

## 1. False-invalid 定义和结论

这里采用的定义：

`false invalid` = 局部 proof obligation 实际 valid，但 GV 最终返回 invalid。

按这个定义，当前汇总中没有 confirmed false-invalid：

| 来源 | 审计范围 | confirmed false-invalid |
|---|---:|---:|
| Positive-control no-baseline | 50 个人工筛过的正确 OPC 局部 step | 0 |
| Missing-premise use case, missing=0 | 10 个完整前提 case | 0 |
| OPC weak / GV-valid audit | 5 个 weak 或 GV-valid case | 0 |
| Noimplicit structured hacker | 13 个 case 的最终状态 | 0 confirmed |

注意：`GV valid` on OPC annotation-invalid 不是 false-invalid。那类是“局部命题 valid，但和 OPC 原 annotation 或全局证明目标不一致”的问题，应标成 local/global mismatch，而不是 GV 误判 invalid。

## 2. OPC weak / GV-valid 人工审计

来源：`codex55_high_opc_stepd_gv_v2_50cases_001`，审计所有 GV valid 或 automatic alignment weak 的 case。

| case | GV | manual label | 人工判断 | 是否 false-invalid |
|---|---|---|---|---|
| `003_3_OPC_best_of_n_BMOSL_2018_12_13` | invalid | `metric_mismatch` | GV 抓到具体 inequality-transfer 错误；annotation 强调 non-sharpness/equality-case。是错误诊断口径不同，不是误判。 | no |
| `004_4_OPC_best_of_n_BMOSL_2019_9_19` | valid | `underformalized_assumption` | 若接受 tangent-chord theorem 接口，局部 Lean step valid；orientation applicability 未独立检查。 | no |
| `005_1_OPC_best_of_n_BMOSL_2019_9_21` | valid | `local_valid_global_valid` | directed angles mod pi 设定下局部有效；annotation 可能按 ordinary angle/supplementary angle 理解。 | no |
| `005_5_OPC_best_of_n_BMOSL_2017_5_25` | valid | `local_valid_global_mismatch` | Lean 证明了 `mth powers` 的局部命题；原题需要 `powers of m`。这是 global bridge/stepD mismatch，不是局部 Lean 错。 | no |
| `010_4_OPC_best_of_n_BMOSL_2017_22_49` | invalid | `metric_mismatch` | GV 给出 convexity maximization 的反例；annotation 说 computational steps skipped。GV 诊断更具体但和 annotation wording 不一致。 | no |

这里最容易误读的是两个 GV invalid 且 alignment 低的 case：

- `003_3...`：不是 false-invalid，因为局部步骤确实 invalid；只是 GV 给出的 invalid reason 与 annotation reason 不同。
- `010_4...`：不是 false-invalid，因为 GV 的反例有效；annotation 只是更粗地描述为计算缺失。

## 3. Positive-control false-invalid 审计

来源：`positive_gv_v2_no_baseline_codex55_high_50_001`

| 指标 | 数值 |
|---|---:|
| total positive-control cases | 50 |
| GV valid | 50 |
| GV invalid | 0 |
| false-invalid case ids | none |

人工解释：这些 case 是从 OPC 中筛出的正确局部 step decomposition。GV 全部返回 valid，说明当前 prompt 在正例上没有过度拒绝。

## 4. Missing-premise 审计

来源：`missing_premise_usecase_gv_v2_codex55_high_10groups_001`

| missing premise count | cases | GV result | 人工判断 |
|---:|---:|---|---|
| 0 | 10 | 10 valid | 正确接受完整局部前提 |
| 1 | 10 | 10 invalid | 正确指出少前提导致结论不可推出 |
| 2 | 10 | 10 invalid | 正确指出少前提导致结论不可推出 |
| 3 | 10 | 10 invalid | 正确指出少前提导致结论不可推出 |

典型判断：

| case | missing | GV 判断 | 人工解释 |
|---|---:|---|---|
| `usecase_g09_integer_additive_chain_missing_0` | 0 | valid | `u=1`, `v=u+1`, `w=v+2` 能推出 `w=4`。 |
| `usecase_g09_integer_additive_chain_missing_1` | 1 | invalid | 少了 `w=v+2`；可取 `u=1,v=2,w=0`，前提成立但结论 `w=4` 不成立。 |
| `usecase_g02_real_order_chain_missing_3` | 3 | invalid | 无足够 order-chain 前提，不能推出最终不等式。 |

这些 invalid 不是 false-invalid，因为测试目标就是模拟用户少给关键 premise。

## 5. Structured hacker 人工审计

### 5.1 pre-fix: `implicit_dependency` 暴露的真实问题

| case | raw status | 人工判断 |
|---|---|---|
| `structured_stepd_hacker_gv_v2_codex55_high_attached_case0_001` | `hack_success_gv_valid` | confirmed miss。hacker 把缺失条件塞进 `implicit_dependency`，Lean 当成可用前提。这个问题推动删除 `implicit_dependency`。 |
| `structured_stepd_hacker_gv_v2_codex55_high_attached_case1_001` | `hack_detected_invalid` | 正确检测。 |
| `structured_stepd_hacker_gv_v2_codex55_high_attached_case2_001` | `hack_success_gv_valid` | false attack。目标从 `∀ y∈[0,1)` 取 `y=0` 和 `y=1/2` 是忠实且有效的局部推理。 |

### 5.2 noimplicit: raw `hack_success_gv_valid` 的人工判读

这些 raw success 都不是 confirmed miss；人工判断为 false attack / benign local step。

| run | raw status | 人工判断 |
|---|---|---|
| `structured_stepd_hacker_gv_v2_noimplicit_codex55_high_case1_001` | GV valid | false attack：局部 reflection involution 存在；hacker 抱怨的是下游接口。 |
| `structured_stepd_hacker_gv_v2_noimplicit_codex55_high_case2_001` | GV valid | false attack：witness/coverage 在局部命题中成立。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset3_001` | GV valid | false attack：选择到 `S1` 外 primes 的稀疏 injection 是有效存在性构造。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset4_001` | GV valid | false attack：`s=1` 的 normalized sum 由上下文/目标支持。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset7_001` | GV valid | false attack：几何分支条件 `F` 在 `AC` 上且 `0≤t≤1` 已明确给出。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset10_001` | GV valid | false attack：hacker 自己的 `why_invalid` 基本承认没有真实 flaw；计算局部有效。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset11_001` | GV valid | false attack：partite hypergraph matching 已包含 coordinatewise disjointness。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset12_001` | GV valid | false attack：目标假设/使用剩余 admissible primes infinite/unbounded，Lean 忠实。 |

### 5.3 noimplicit: GV detected invalid 的人工判读

这些 case 是 true detection，不是 false-invalid。

| run | invalid reason | 人工判断 |
|---|---|---|
| `structured_stepd_hacker_gv_v2_noimplicit_codex55_high_case0_001` | derivative sign / uniform eta / permutation-prefix 等多轮攻击 | true detection。典型首轮：由 `0≤t≤1-a` 推 `1-2t/(1-a)≥0` 是假的。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset5_001` | finite branch coverage / arithmetic growth / simultaneous ordering | true detection。最终重要例子：枚举 `15≤n≤36` 的奇合数遗漏 `25`。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset8_001` | algebraic normalization / missing triangle inequalities | true detection。例如 `1/(ab)+1/(bc)+1/(ca)` 的分母写成 `ab+bc+ca` 是错误恒等式。 |
| `structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset9_001` | witness uniformity / gcd after addition | true detection。例如“独立构造所以可令 `d=c`”不成立。 |

`structured_stepd_hacker_gv_v2_noimplicit_10more_codex55_high_offset6_001` 是 hacker gave up，未形成有效攻击。

## 6. 当前人工审计口径

以后建议报告三个分开的标签：

| label | 含义 |
|---|---|
| `local_valid` | 给定 stepD 的局部 proposition 在自身 premises 下成立。 |
| `global_bridge_mismatch` | 局部 proposition 成立，但不是原题/原证明真正需要检查的命题。 |
| `confirmed_false_invalid` | 局部 proposition 成立，但 GV 返回 invalid。当前主实验中没有确认案例。 |
