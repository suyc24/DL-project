# Adaptive Adversarial GV 研究报告

日期：2026-06-04
项目目录：`/root/DL-project/lean_single_step_formalization`

## 1. 摘要

本轮工作围绕一个核心问题展开：能否构造或筛选出自然语言 verifier 初判无法正确识别、但 Lean/GV 局部形式化能够暴露错误的数学伪证样本。

结论比较清楚：

1. 直接使用 OPC 已标注错证时，当前强 baseline 非常强。最终 50 条公平对照中，baseline 判 invalid 为 49/50，GV 判 invalid 也是 49/50；baseline annotation alignment 平均 8.50/10，GV 为 8.20/10。
2. Lean/GV 确实能提供一种重要信号：generator 为了让局部 theorem 编译，常常必须引入强假设、局部接口或接近最终结论的 premise；review 可以据此判断自然语言 step 是否缺条件或不忠实。但这个信号并不自动成立，需要 verifier 正确区分“原证明真的使用了一个局部假设”和“generator 为了过 Lean 塞入了不忠实前提”。
3. 局部 Lean theorem 的 `compile_ok=true` 不能解释为原命题被验证。很多 Lean 文件证明的是“在强假设下的局部结论”，不是题目原命题或自然语言证明的真正推导。
4. `step_d` 是目前最关键的改进：它把目标 proof step 结构化为 premise/proof step/conclusion，同时补出中性 implicit dependencies。它显著改善了 GV 对隐藏依赖的识别，但也会增强 baseline，因此公平比较时 baseline 也应拿到相同 `step_d`。
5. adversarial hacker 很难生成真正绕过强 baseline 的伪证。历史 run 中大量结果是 `too_obvious` 或 `hacker_failed`；即使给 hacker 明确要求“强文本 verifier 可能初判 valid，但 Lean 更容易暴露”，它仍经常生成 baseline 一眼可抓的错误。

因此，当前系统更适合作为“proof obligation / faithfulness auditor”，而不是已经能在现有 OPC slice 上稳定显著击败强文本 baseline 的完整 benchmark。

## 2. 代码与流程概览

### 2.1 主要脚本

- `scripts/run_adversarial_game.py`
  - 早期 adaptive adversarial game。
  - 流程为 hacker 生成 modified CoT，baseline 与 lean-assisted 判断。
  - 分类包括 `too_obvious`、`lean_rescue`、`lean_missed`、`lean_weaker_than_baseline`。

- `scripts/run_adversarial_game_gv.py`
  - 当前 GV 主流程。
  - 将 lean-assisted 拆成 generator/verifier 两个角色。
  - generator 写 Lean 文件并运行 `lake env lean <file>`。
  - verifier review generator 的 Lean 文件、编译结果和自然语言 step。

- `scripts/run_gv_steps_probe.py`
  - 直接对已有 single-step rows 跑 GV，不经过 hacker。
  - 早期用于 OPC 单例、trusted initial 和 prompt 测试。

- `scripts/run_initial_filter.py`
  - 对候选 step 跑 initial verifier。
  - 后来用于筛选 baseline 初判 valid 的候选。

- `scripts/run_opc_llm_scope_batches.py`
  - 对 OPC annotation 选择更大的自然语言 scope。
  - 这是为了解决早期“scope 太窄，目标 step 中看不出错误”的问题。

- `scripts/run_stepd_gv_alignment_batches.py`
  - 最终主要批量 runner。
  - 每个 case 跑：
    - `step_d`
    - baseline：`verifier_initial` + `step_d`
    - our method：`step_d -> GV`，跳过 initial，直接跑 generator/review
    - annotation alignment scoring
  - 支持 5 个一组并行，支持 repair round。

- `scripts/run_hacker_stepd_gv_game.py`
  - 本轮新增。
  - 将 hacker 生成的 adversarial rows 接到 `step_d -> baseline/GV` 公平对照链路。
  - 目标分类仍然是 `baseline != invalid && GV == invalid`。
  - 目前只完成了 smoke 和少量中断/单轮测试，结果显示 hacker 仍容易生成 too-obvious 错误。

### 2.2 主要 prompt

- `prompts/adaptive_adversarial_gv/verifier_initial.md`
  - 当前 baseline initial 很强：
    - 判断对象是目标步骤本身是否严格可靠。
    - 只有完全正确才能 valid。
    - 推理细节不对、缺关键理由、用词不准确均 invalid。

- `prompts/adaptive_adversarial_gv/step_decompose.md`
  - 当前 step_d prompt。
  - 目标是中性结构化 proof step，不输出 verdict。
  - 关键设计：
    - human annotation 只用于提醒概念保留，不暴露 gold/error/invalid。
    - 不修正原文，不替换概念。
    - 但要列出原 step 默认依赖的 hidden assumptions / theorem applicability / construction obligations。
    - 对构造性 step，必须拆出表示形式、节点条件、系数条件、次数/非零首项、整除条件等 proof obligations。

- `prompts/adaptive_adversarial_gv/generator_formalize.md`
  - 当前 generator 被定义为“局部 Lean 定理转换器”。
  - 重要规则：
    - Lean 不一定要过，忠实比编译成功更重要。
    - theorem 的每个前提必须有自然语言来源，或明确标为 step 需要但未证明的局部依赖。
    - 不要把完整结论原样当前提。
    - 不要用粗粒度 lemma 包装完整结论。

- `prompts/adaptive_adversarial_gv/verifier_review.md`
  - 当前 review 是三分类：
    - `return_valid`
    - `request_repair`
    - `return_invalid`
  - 核心策略：
    - Lean 与自然语言地位平等。
    - 逐字/逐字段对应对象、关系、指数、量词、范围。
    - 如果 Lean 基本表达了自然语言，但自然语言本身缺条件/偷换/证明细节错误，则 `return_invalid`。
    - 如果主要是 Lean 没有表达自然语言，则 `request_repair`。

### 2.3 sandbox/bwrap 问题

早期 Codex/Lean 子进程出现过：

```text
bwrap: Failed to make / slave: Permission denied
```

后续运行统一使用 `--codex-sandbox danger-full-access`，并在 `run_adversarial_game_gv.py` 中通过 `agent_sandbox(provider, sandbox)` 避免在 Codex 子 agent 中使用 `read-only` / `workspace-write`。这使 generator 能直接在 `/root/mathlib4/.single_step_gv_workspaces/...` 写 Lean 文件并运行 `lake env lean`。

## 3. 研究流程演化

### 3.1 早期 adaptive hacker + baseline/GV

早期目标是让 hacker 从候选 CoT 中生成伪证，然后比较 baseline 与 Lean-assisted。

代表 run：

| Run | Cases | Rounds | lean_rescue | too_obvious | lean_missed | lean_weaker | hacker_failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `codex_adaptive_3_max10_modified_cot` | 3 | 13 | 0 | 10 | 1 | 2 | 0 |
| `gv_10x10_latest` | 10 | 13 | 0 | 10 | 0 | 1 | 1 |
| `gv_10x10_fair_baseline` | 10 | 20 | 0 | 16 | 1 | 0 | 3 |
| `gv_harder_olympiadbench10_newprompt_10x10` | 10 | 17 | 0 | 12 | 0 | 0 | 5 |
| `gv_harder_olympiadbench10_fair` | 10 | 8 | 0 | 6 | 1 | 0 | 0 |

观察：

- hacker 经常生成太明显的错误，baseline 和 GV 都能抓住。
- 有时 Lean-assisted 比 baseline 弱，说明初期 GV prompt/repair/review 设计不稳定。
- `model_rescue_no_lean` 出现过，表示 lean-assisted initial 自己判 invalid，并不是 Lean 贡献。
- 真正 `baseline != invalid && Lean/GV invalid` 的目标样本很少，基本没有稳定出现。

### 3.2 OPC 数据进入后：scope 是关键

OPC 数据提供了 human annotation 的错误证明 step，但直接用 annotation 的窄 span 经常不够。典型问题：

- annotation 只标了一个短短的 phrase。
- 单看 phrase 可能并没有错。
- 错误必须结合前后 proof context 才能看出。

因此加入了 LLM scope selection：

- `scripts/run_opc_llm_scope_batches.py`
- 相关数据：
  - `experiments/runs/opc_llm_scope_candidate_filter_allann_50x5_*/steps/group_*.jsonl`

这一步解决了“目标 step 太窄”的问题，但也带来新问题：scope 扩大后，baseline 能看到更多上下文，也更容易直接判 invalid。

### 3.3 step_d 引入

用户指出：应由一个模块把自然语言 step 结构化成前提 list、证明 step list、最终结论，再喂给 GV。

于是加入 `step_decompose.md` 与 `run_step_decomposition_for_row`。

设计目标：

- 让后续 GV 不只看原句，而能看到 proof step 中实际使用的依赖。
- 但不能告诉 GV “这里有错”，否则相当于作弊。
- human annotation 只能用于提醒 step_d 保留关键概念，不可输出 gold/error/wrong/invalid。

后续 prompt 迭代中，一个重要纠偏是：

- implicit dependency 不是暴露 annotation 中的错误。
- implicit dependency 是 proof step 若要成立所需要的所有默认前提。
- 例如构造性 step 中要补出 `h_values`、`h_degree`、`h_lagrange` 这些关键结论作为局部 proof obligations，而不是写出“m-th powers 与 powers of m 不同”。

### 3.4 review prompt 从 pair schema 改为三分类

早期 `verifier_review` 要求输出 Lean/NL pair 对应字段，后来发现这会让 review 负担过重，且不稳定。

当前策略：

- 备份旧 pair prompt 到 `verifier_review.backup_pair_schema.md`。
- 新 review 只判断三类：
  - Lean/NL 对应且共同支持 step：valid。
  - Lean 不忠实：repair。
  - 自然语言 step 本身错误或缺条件：invalid。

关键原则：

- 不再单方面问“Lean 是否忠实对应自然语言 proof step”。
- Lean 和自然语言平权。
- 要判断到底是哪边缺条件、多条件、证明了别的命题。

### 3.5 annotation alignment 打分

为了比较 returned invalid reason 是否真的对齐 human annotation，加入：

- `prompts/adaptive_adversarial_gv/annotation_alignment.md`
- `score_alignment(...)`

评分 0-10：

- 0：没有 invalid reason 或完全不相关。
- 10：returned reason 与 human annotation 的核心错误高度一致。

这是一个重要指标，因为简单的 invalid rate 不够。例如某个 verifier 可以判 invalid，但理由是泛泛的“证明缺口”，并没有抓到 annotation 中真正的偷换。

## 4. 主要实验结果

### 4.1 初始 25 条 OPC step_d/GV alignment

Run：`experiments/runs/opc_stepd_gv_alignment_50cases_001`

该 run 实际完成 25 条后因 weak 过多停止。

汇总：

| Metric | Value |
|---|---:|
| cases | 25 |
| baseline invalid | 24/25 |
| GV invalid | 22/25 |
| weak | 6/25 |
| baseline alignment avg | 7.92/10 |
| GV alignment avg | 6.92/10 |

weak case：

- `002_2_OPC_best_of_n_USAMO_2015_5_7`
- `003_3_OPC_best_of_n_BMOSL_2018_12_13`
- `004_4_OPC_best_of_n_BMOSL_2019_9_19`
- `005_1_OPC_best_of_n_BMOSL_2019_9_21`
- `005_4_OPC_best_of_n_BMOSL_2017_5_24`
- `005_5_OPC_best_of_n_BMOSL_2017_5_25`

主要问题：

- GV review 有时把 Lean 的强 premise 当作可接受局部接口。
- step_d 对构造性 step 的默认依赖列得不够细。
- review 没有稳定对齐原题短语与 target step 短语。

### 4.2 weak retest 系列

#### Retest 001

Run：`opc_stepd_gv_alignment_weak_retest_001`

| Metric | Value |
|---|---:|
| cases | 6 |
| baseline invalid | 5/6 |
| GV invalid | 3/6 |
| weak | 6/6 |
| baseline avg | 4.17 |
| GV avg | 1.00 |

说明当时 review prompt 仍不稳定，GV 甚至比 baseline 弱很多。

#### Retest 002

Run：`opc_stepd_gv_alignment_weak_retest_002`

| Metric | Value |
|---|---:|
| cases | 6 |
| baseline invalid | 5/6 |
| GV invalid | 4/6 |
| weak | 4/6 |
| baseline avg | 4.83 |
| GV avg | 3.17 |

改进来自：

- review prompt 更强调 Lean 文件内容。
- Lean/NL 对照更平权。
- request_repair / return_invalid 边界更清晰。

#### Retest 003

Run：`opc_stepd_gv_alignment_weak_retest_003`

| Metric | Value |
|---|---:|
| cases | 6 |
| baseline invalid | 4/6 |
| GV invalid | 5/6 |
| weak | 3/6 |
| baseline avg | 2.83 |
| GV avg | 4.17 |

这一版引入了更强的构造性 implicit dependency 拆解：

- 对 Lagrange interpolation、integer coefficients、degree exactly `n`、nonzero top coefficient 等 obligation 分开列出。
- 对 divisibility/constructive proof step 不再只写“will construct”。

效果：

- `BMOSL_2017_5_24` 从 old GV valid/0 改到 new GV invalid/10。
- `BMOSL_2019_9_19` 从 old GV valid/0 改到 new GV invalid/7 或 8。
- `USAMO_2015_5_7` 从 weak proof-gap 改到非 weak。

但仍然有两个长期问题：

- angle/tangent-chord 类 step 在有向角 mod pi 下有 annotation/context ambiguity。
- `BMOSL_2017_5_25` 中 `perfect powers of m` vs `m-th powers` 的核心语义 mismatch 仍未被 GV review 抓住。

### 4.3 最终 50 条 OPC 公平对照

Run：`experiments/runs/opc_stepd_gv_alignment_50cases_construct_obligations_001`

命令配置：

```bash
python scripts/run_stepd_gv_alignment_batches.py \
  --run-id opc_stepd_gv_alignment_50cases_construct_obligations_001 \
  --max-cases 50 \
  --batch-size 5 \
  --parallel-cases 5 \
  --repair-rounds 3 \
  --codex-sandbox danger-full-access \
  --stop-weak-threshold 999
```

总体结果：

| Metric | Value |
|---|---:|
| cases | 50 |
| baseline invalid | 49/50 |
| GV invalid | 49/50 |
| Lean used | 50/50 |
| weak | 2/50 |
| baseline annotation alignment avg | 8.50/10 |
| GV annotation alignment avg | 8.20/10 |
| GV better than baseline | 7 |
| equal | 29 |
| GV worse than baseline | 14 |

分组结果：

| Group | Cases | Baseline invalid | GV invalid | Baseline avg | GV avg | Weak |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 5 | 5 | 5 | 9.60 | 9.60 | 0 |
| 002 | 5 | 5 | 5 | 8.80 | 8.60 | 0 |
| 003 | 5 | 5 | 5 | 6.40 | 7.00 | 0 |
| 004 | 5 | 5 | 5 | 9.60 | 8.80 | 0 |
| 005 | 5 | 4 | 4 | 5.60 | 5.40 | 2 |
| 006 | 5 | 5 | 5 | 9.20 | 8.80 | 0 |
| 007 | 5 | 5 | 5 | 9.60 | 7.40 | 0 |
| 008 | 5 | 5 | 5 | 8.40 | 8.60 | 0 |
| 009 | 5 | 5 | 5 | 9.40 | 9.20 | 0 |
| 010 | 5 | 5 | 5 | 8.40 | 8.60 | 0 |

最终 weak cases：

1. `005_1_OPC_best_of_n_BMOSL_2019_9_21`
   - baseline: `valid`, alignment 0/10
   - GV: `valid`, alignment 0/10
   - 两者都接受了 tangent-chord angle step。

2. `005_5_OPC_best_of_n_BMOSL_2017_5_25`
   - baseline: `invalid`, alignment 2/10
   - GV: `invalid`, alignment 2/10
   - GV invalid reason 主要围绕 Lagrange 插值不能自动保证整数系数和次数正好 `n`，没有抓住 annotation 核心：把 “perfect powers of `m`” 误读为 “`m`-th powers”。

解释：

- 这个 50-case slice 不支持“GV 显著优于强 baseline”的结论。
- 它支持一个更保守的结论：在同样拿到 `step_d` 的情况下，GV 能稳定暴露 hidden obligations，但 baseline 本身也能大量抓错。
- 如果目标是证明 Lean/GV 的相对优势，需要更难的数据，尤其是 baseline 初判 valid 的样本。

## 5. 案例分析

### 5.1 正例：BMOSL 2017/5 的 Lean faithfulness rescue

文档：`docs/opc_bmosl_2017_5_lean_rescue_example.md`

Case：`OPC_best_of_n_BMOSL_2017_5_3`

自然语言目标 step：

```text
Proof. We will construct by Lagrange-interpolation a polynomial of degree exactly n
whose values at 0,1,...,n are prescribed mth powers, and we will arrange the
interpolating coefficients to lie in Z.
```

OPC diagnosis：

```text
Misunderstands the problem: instead of powers of m, interpolates m-th powers.
```

早期 weak initial 设置下：

- baseline initial: `valid`
- lean-assisted verifier initial: `valid`
- GV final: `invalid`
- status: `lean_rescue`

生成的 Lean 本质上是：

```lean
theorem step_contract ...
    (constructed :
      ∃ P : Polynomial ℤ,
        P.natDegree = n ∧
          (∀ k : ℕ, k ≤ n → ∃ a : ℤ, P.eval (Int.ofNat k) = a ^ m)) :
    ∃ P : Polynomial ℤ,
      P.natDegree = n ∧
        (∀ k : ℕ, k ≤ n → ∃ a : ℤ, P.eval (Int.ofNat k) = a ^ m) := by
  exact constructed
```

这个 Lean 编译通过，但 review 正确指出：

- `constructed` 几乎就是最终存在性结论。
- Lean 没有表达或验证 Lagrange 插值。
- Lean 没有表达或验证插值系数属于 `ℤ`。
- Lean 没有证明 degree exactly `n`。

这是一个有价值的 Lean/GV 信号：不是 Lean 编译失败，而是 Lean 为了编译必须把核心构造作为 premise。

但是这个正例也说明方法的脆弱性：

- 这里的 rescue 依赖 weak/trusted initial 设置。
- 在强 baseline + step_d 下，baseline 往往也能抓出同类问题。

### 5.2 弱例：`perfect powers of m` vs `m-th powers`

Case：`005_5_OPC_best_of_n_BMOSL_2017_5_25`

路径：

```text
experiments/runs/opc_stepd_gv_alignment_50cases_construct_obligations_001/
  groups/group_005/005_5_OPC_best_of_n_BMOSL_2017_5_25/
```

Lean 编译结果：

```json
{
  "compile_ok": true,
  "stdout_tail": "",
  "stderr_tail": ""
}
```

实际 Lean 片段：

```lean
def IsPrescribedMPowerData (m n : ℕ) (values : Fin (n + 1) → ℤ) : Prop :=
  ∀ i : Fin (n + 1), ∃ a : ℤ, values i = a ^ m
```

这表示 “`m`-th powers”，而不是题目要求的 “powers of `m`”。后者应更接近：

```lean
∀ i, ∃ k, values i = m ^ k
```

该 Lean theorem 的核心前提包括：

```lean
h_lagrange :
  ... → ∃ P : Polynomial ℤ, LagrangeInterpolationOutput m n values P

h_integral_coefficients :
  ∀ P, LagrangeInterpolationOutput m n values P → HasIntegerCoefficients P

h_top :
  ∀ P, LagrangeInterpolationOutput m n values P → TopCoefficientNonzero P n
```

这说明它能过编译，是因为：

1. 它把 target step 中的错误短语 `mth powers` 忠实翻译成了 `a ^ m`。
2. 它把困难的构造义务作为强假设传入。
3. theorem 只证明“在这些强假设下，存在所需的局部对象”。

GV 最终判 invalid，但理由是：

- Lagrange interpolation 不能自动保证 degree exactly `n`。
- 不能自动保证 integer coefficients。
- Lean 依赖 `h_integral_coefficients` 和 `h_top`。

这确实是一个证明缺口，但没有对齐 human annotation 的核心 mismatch。因此 alignment 只有 2/10。

这个案例是当前方法最重要的反例之一：

- `compile_ok=true` 不代表原题被验证。
- generator 可以把错误自然语言 step 形式化成一个合法局部 theorem。
- review 必须判断 Lean theorem 是：
  - 忠实暴露了自然语言 step 的隐藏依赖；
  - 还是为了过 Lean 而选择了不忠实/偏题的形式化；
  - 或者自然语言 step 本来就在使用一个错误概念。

这个三者区分非常难，尤其在局部 theorem 本身是条件化的情况下。

### 5.3 修复成功例：BMOSL 2017/5/24 的整除义务

Case：`005_4_OPC_best_of_n_BMOSL_2017_5_24`

旧结果：

- 初始 25-case run 中 GV weak，甚至曾判 valid。

最终 50-case run：

- baseline alignment: 9/10
- GV alignment: 10/10
- GV verdict: invalid

GV reason 核心：

```text
目标步骤依赖“已经安排 k!(n-k)! ∣ m^{A_k}”。
但若 m 不含 n! 的某些素因子，则任何 m^C 都不可能被 (n!)^2 整除。
例如 m=2,n=3 时，(3!)^2=36 含素因子 3，而 2^C 永远不被 3 整除。
```

这个提升来自 step_d 对 implicit dependencies 的改进：

- 不再只写“将安排整除”。
- 明确列出 divisibility obligation。
- 使 GV review 能看到 Lean 中哪些 premise 承载了无法从题设推出的关键整除断言。

### 5.4 不等式方向例：BMOSL 2018/12/13

Case：`003_3_OPC_best_of_n_BMOSL_2018_12_13`

最终 50-case run：

- baseline alignment: 2/10
- GV alignment: 4/10
- GV verdict: invalid

GV reason：

```text
设 S = PB·PC + PC·PA + PA·PB。
由 (2.1) 得 S > k^2(xy+yz+zx)，由 (2.2) 得 2Δ ≤ S。
这两条只能说明 2Δ 不超过一个大于 k^2(...) 的量，
不能推出 2Δ > k^2(...).
```

这个例子展示了 GV 可以比 baseline 更具体地解释中间量关系，但 alignment 仍偏低，因为 annotation 与 returned reason 的粒度不完全一致。

### 5.5 几何角度例：BMOSL 2019/9/21

Case：`005_1_OPC_best_of_n_BMOSL_2019_9_21`

最终结果：

- baseline: valid, alignment 0/10
- GV: valid, alignment 0/10
- weak: true

双方都认为：

- 在 `X,Y,D,H` 已在圆 `Γ` 上、
- `AX, AY` 是切线、
- 使用 directed angles modulo pi 的情况下，
- 两次 tangent-chord theorem 可推出目标角等式。

这类几何 case 的问题是：

- annotation 可能依赖具体图形取向、弧选择或普通角/有向角约定。
- 如果上下文已经明确 directed angles mod pi，某些“补角”问题会消失。
- baseline 和 GV 都可能把标准几何事实视为可接受。

因此它可能不是纯粹的系统失败，也可能是 annotation/scope/角度约定的不一致。

### 5.6 Hacker-on-OPC 例：仍然 too obvious

新增 runner：`scripts/run_hacker_stepd_gv_game.py`

OPC hacker run：

- `opc_hacker_stepd_gv_5x3_001`
- `opc_hacker_stepd_gv_5x1_001`

由于用户中断，两个 run 都只完成 1 round。

结果：

| Run | Rounds | Status |
|---|---:|---|
| `opc_hacker_stepd_gv_5x3_001` | 1 | `too_obvious` |
| `opc_hacker_stepd_gv_5x1_001` | 1 | `too_obvious` |

代表攻击：

```text
若 1+i | gcd(α,β)，则 2 | e，从而 32 | e^5=a^4+b^4。
但当一个第四幂和能被 32 整除时，两个底数必须同为偶数；
把 a,b 同除以 2 后仍得到同类表示，这与取 a^4+b^4=e^5 为最小的表示矛盾。
```

baseline 直接抓住：

```text
把 a,b 同除以 2 后仍得到同类表示不成立：
(a/2)^4+(b/2)^4=e^5/16；若 e=2e_1，则右端为 2e_1^5，通常不是五次幂。
此外“取最小表示”并非题目或前文已建立前提。
```

这说明即使使用 OPC 题目继续让 hacker 生成新伪证，当前 hacker 仍容易生成强 baseline 能直接识别的错误。

## 6. 核心方法论问题

### 6.1 Lean 编译通过不等于数学验证

最重要的发现是：

```text
compile_ok=true
```

只说明 Lean theorem 在它自己的 definitions 和 assumptions 下成立。

它不说明：

- definitions 忠实表达了题目。
- assumptions 都来自题目/前文/target step。
- theorem 没有把最终结论或关键中间结论作为 premise。
- formalization 没有替换概念、量词、方向、指数含义。

因此，当前 GV 的核心不是 Lean compile check，而是 faithfulness review。

### 6.2 局部 theorem 的强假设歧义

局部 Lean theorem 必然需要一些 local assumptions。问题是：

```text
哪些 local assumptions 是原 proof step 合理调用的上下文接口？
哪些 local assumptions 是 generator 为了让 theorem 过 Lean 而加入的不忠实假设？
哪些 local assumptions 正好暴露自然语言 proof step 本身缺少证明义务？
```

这三者非常难区分。

例子：

- `h_lagrange`：可能是“标准 Lagrange 插值定理”的合理接口。
- `h_integral_coefficients`：通常不是标准 Lagrange 插值直接给出的，需要额外整除构造。
- `h_top`：用于 degree exactly `n`，通常需要证明最高项非零。
- `constructed`：几乎就是最终结论本身，应视为不忠实。
- `IsPrescribedMPowerData`：虽然忠实于 target step 的 “mth powers”，但不忠实于原题 “powers of m”。

所以 review 不应只问“Lean theorem 能不能编译”，而必须维护 assumption ledger。

### 6.3 step_d 是双刃剑

step_d 的好处：

- 显著提高 GV 对隐藏义务的可见性。
- 让 generator 更容易产出结构化局部 theorem。
- 让 review 能逐字段检查 Lean premise 的来源。

step_d 的风险：

- 它也增强 baseline。
- 如果写得太接近 annotation，会变成作弊。
- 如果 implicit dependency 写出“两个短语不等价”，就直接泄露 gold error。
- 如果 implicit dependency 过于中性，又可能无法帮助 review 抓核心 mismatch。

当前策略是：

- 保留题目短语与 target step 短语的逐字差异。
- 不主动写出二者等价/不等价。
- 让后续 verifier 自己发现。

`BMOSL_2017_5_25` 说明这个策略仍不够：step_d 保留了 “perfect powers of m” 和 “mth powers”，但 review 没抓住。

### 6.4 baseline 很强，不能人为调差

用户明确指出过：把 baseline 或 verifier initial prompt 调差会有作弊风险，因为真实 baseline 可以更强。

最终公平设置中，baseline 拿到和 GV 相同的 `step_d`，结果：

- baseline invalid 49/50
- GV invalid 49/50
- baseline avg 8.50
- GV avg 8.20

这说明：

- 不能通过弱化 baseline 来制造优势。
- 需要找到 baseline 真正难以发现的样本。
- 当前 OPC slice 很可能太容易，或 annotation 错误过于显性。

### 6.5 hacker 难以突破强 baseline

历史 adversarial hacker 结果显示：

- 大量 `too_obvious`
- 不少 `hacker_failed`
- 几乎没有稳定 lean_rescue

原因可能包括：

- hacker 仍倾向于简单必要/充分偷换、边界错误、常数/公式错误。
- prompt 虽要求隐蔽错误，但生成模型会优先产出自然语言可解释的错误。
- 一旦 baseline prompt 要求严格检查目标步骤每个条件和用词，许多伪证都会被直接抓住。

这不是单纯 prompt 问题，也可能说明需要更系统的攻击生成方法。

## 7. 数据与实验可信度

### 7.1 公平性较好的部分

最终 50-case run 公平性较好：

- baseline 和 GV 都使用相同 `step_d`。
- GV 跳过 initial，直接运行 Lean/GV。
- repair round 设为 3。
- 每条都有 annotation alignment score。

但仍有局限：

- alignment scoring 本身由 LLM 完成，可能有评分噪声。
- OPC annotations 可能有 scope/角度约定不一致。
- allann_50x5 step selection 不一定代表最难样本。

### 7.2 探索性结果不能过度解读

以下结果不能作为最终方法优势证据：

- weak initial 下的 `lean_rescue`。
- trusted initial 下的 `lean_rescue=5/5`。
- `model_rescue_no_lean`。
- 旧 parser 只看窗口时得到的“骗过 baseline/GV”结果。

它们仍然有价值，因为帮助发现：

- scope 问题；
- generator 可以直接退出 invalid 的副作用；
- review prompt 的不稳定；
- pair schema 过于复杂；
- step_d 的必要性。

## 8. 建议的下一步

### 8.1 建立 assumption ledger

每个 Lean theorem 的 premise 应该被显式分类：

| 类型 | 含义 |
|---|---|
| problem/context | 题目或前文明确给出 |
| target_claim | target step 明确声称 |
| standard_theorem | 可接受标准定理接口 |
| proof_obligation | target step 需要但没有证明的义务 |
| generator_added | generator 自行添加，缺少自然语言来源 |
| conclusion_smuggling | 近似最终结论或关键中间结论被当前提 |

review 不应只输出 reason，而应输出 premise ledger。指标可以包括：

- unsupported premise count
- conclusion-smuggling count
- proof-obligation count
- semantic mismatch count

### 8.2 分离三个任务

当前 GV 混合了三个判断：

1. Lean theorem 是否编译。
2. Lean theorem 是否忠实表达自然语言 step。
3. 自然语言 step 是否在原题/前文下严格可靠。

建议拆成三个显式输出：

```json
{
  "compile_status": "...",
  "faithfulness_status": "...",
  "mathematical_validity_status": "...",
  "unsupported_assumptions": [...],
  "semantic_mismatches": [...]
}
```

这样可以避免 `compile_ok=true` 被误读，也能更清楚地区分：

- Lean 不忠实，需要 repair；
- Lean 忠实，但自然语言 step 错；
- Lean 忠实且自然语言 step 可接受。

### 8.3 针对术语做 glossary pass

`perfect powers of m` vs `m-th powers` 说明，在进入 Lean 前应先做术语表：

```json
{
  "term": "perfect powers of m",
  "source": "problem",
  "candidate_formal_meaning": "∃ k, value = m^k",
  "confusable_with": "m-th powers: ∃ a, value = a^m"
}
```

但 glossary 也不能直接告诉 verifier “这里错了”。合理做法是：

- 列出题目短语的候选 formal meaning。
- 列出 target step 短语的候选 formal meaning。
- 要求 review 判断二者是否同义。

### 8.4 构造更难 benchmark

OPC 已标注错证太容易被强 baseline 抓住。更有价值的数据应满足：

- baseline with step_d 初判 valid 或低置信 valid。
- human annotation 确认 invalid。
- 错误不是一眼可见的公式/边界/必要充分问题。
- 错误依赖隐藏 witness、参数独立性、定义域、非零性、局部到全局接口、构造义务。

可以用两阶段筛选：

1. 大规模生成/收集候选。
2. 强 baseline 过滤，只保留 baseline 未判 invalid 的样本。
3. 再跑 GV。

这比人为调弱 baseline 更公平。

### 8.5 改进 hacker

当前 hacker 难以突破 baseline。可考虑：

- hacker 先生成 proof obligation graph，再选择一个不显眼 obligation 删除。
- hacker 不直接改目标 step，而是改前文中某个 dependency 的接口，使目标 step 表面仍自然。
- hacker 用 baseline feedback 做 rejection sampling：每轮先内部模拟 baseline，只有 baseline 可能 valid 才输出。
- 使用 templates：
  - uniform witness vs pointwise witness；
  - parameter fixed vs parameter depending on object；
  - `degree ≤ n` vs `degree = n`；
  - rational coefficients vs integer coefficients；
  - local construction satisfying nodes vs global object satisfying theorem；
  - theorem applicability missing nonzero/positivity/domain conditions。

### 8.6 报告指标建议

后续每个 run 应记录：

- invalid rate
- alignment average
- weak count
- baseline vs GV better/equal/worse
- Lean compile rate
- Lean used rate
- request_repair count
- repair success count
- unsupported premise count
- conclusion-smuggling count
- semantic mismatch count

其中后四项比单纯 compile rate 更能体现 Lean/GV 的价值。

## 9. 当前结论

这轮工作最有价值的结论不是“Lean/GV 已经显著打败 baseline”，而是：

1. 强 baseline 在当前 OPC slice 上非常强，不能通过调差 prompt 来制造结果。
2. 局部 Lean verification 的核心难点是 faithfulness，不是 compilation。
3. 带强假设的局部 Lean theorem 极易混淆：
   - 它可能是合理局部接口；
   - 也可能是 generator 为了过 Lean 加的前提；
   - 还可能正好暴露自然语言 proof step 本身偷用了未证明义务。
4. `step_d` 是必要组件，但必须严格保持中性，不能把 annotation error 直接暴露给 GV。
5. annotation alignment 是必要指标，因为 invalid 本身不够；理由是否对齐 human diagnosis 才是关键。
6. adversarial hacker 目前不是可靠数据来源：它很难处理掉强 baseline，常生成 too-obvious 攻击。

因此，下一阶段最合理方向是：

- 保留强 baseline；
- 建立 assumption ledger 与术语 glossary；
- 用强 baseline 过滤出真正难样本；
- 让 GV 的主要贡献从“Lean 编译”转向“局部 theorem faithfulness / unsupported assumption auditing”。

## 10. 关键文件索引

代码：

- `scripts/run_adversarial_game.py`
- `scripts/run_adversarial_game_gv.py`
- `scripts/run_gv_steps_probe.py`
- `scripts/run_initial_filter.py`
- `scripts/run_opc_llm_scope_batches.py`
- `scripts/run_stepd_gv_alignment_batches.py`
- `scripts/run_hacker_stepd_gv_game.py`

Prompt：

- `prompts/adaptive_adversarial_gv/verifier_initial.md`
- `prompts/adaptive_adversarial_gv/step_decompose.md`
- `prompts/adaptive_adversarial_gv/generator_formalize.md`
- `prompts/adaptive_adversarial_gv/generator_repair.md`
- `prompts/adaptive_adversarial_gv/verifier_review.md`
- `prompts/adaptive_adversarial_gv/annotation_alignment.md`

文档：

- `docs/adaptive_adversarial_flow.md`
- `docs/generator_verifier_lean_assist.md`
- `docs/opc_bmosl_2017_5_lean_rescue_example.md`

主要 run：

- `experiments/runs/gv_10x10_latest`
- `experiments/runs/gv_10x10_fair_baseline`
- `experiments/runs/gv_harder_olympiadbench10_newprompt_10x10`
- `experiments/runs/opc_group001_step_decompose_gv_test`
- `experiments/runs/opc_group001_5_ours_stepd_gv_trusted_initial`
- `experiments/runs/opc_stepd_gv_alignment_50cases_001`
- `experiments/runs/opc_stepd_gv_alignment_weak_retest_001`
- `experiments/runs/opc_stepd_gv_alignment_weak_retest_002`
- `experiments/runs/opc_stepd_gv_alignment_weak_retest_003`
- `experiments/runs/opc_stepd_gv_alignment_50cases_construct_obligations_001`
- `experiments/runs/opc_hacker_stepd_gv_5x3_001`
- `experiments/runs/opc_hacker_stepd_gv_5x1_001`
