# AG-SFV Data Generation

本文档是 `data_generation/` 下数据采集、FHIS 标注、hidden-state feature 和 probe 结果的统一入口。当前主数据 bundle 是：

```text
data_generation/qwen25_fhis/
```

目标是为 Activation-Gated Selective Formal Verification, AG-SFV, 构造一个可以直接训练 step-level probe 的数据集。probe 输入是数学推理每个 step 结束位置的 hidden states，输出是该 step 是否为第一处关键错误，或是否值得优先送去外部验证。

## 1. 当前结论

当前 pipeline 已经闭环：

| 项目 | 当前状态 |
|---|---:|
| 题目 | OlympiadBench `OE_TO_maths_en_COMP` 全量 674 题 |
| 生成模型 | `Qwen/Qwen2.5-Math-7B-Instruct` |
| 生成 trace | 2696 |
| Codex 已标注 completed trace | 2343 |
| Codex-clean 训练 trace | 2128 |
| wrong FHIS trace | 1192 |
| correct negative trace | 936 |
| step hidden-state rows | 8423 |
| FHIS positive rows | 1192 |
| negative rows | 7231 |
| probe hidden logistic AUROC | 0.862 |
| probe hidden logistic AUPRC | 0.545 |
| probe top 30% budget coverage | 0.901 |

`rough_final_correct` 只作为诊断字段。训练标签以本地 Codex 标注为准。

## 2. 为什么选这个数据设置

第一阶段不是追求模型最高正确率，而是需要稳定地产生足够多的错误推理。若题目太简单，wrong trace 太少；若题目太难，CoT 容易退化成格式崩坏或乱猜。

官方正确率参考来自 Qwen2.5-Math Technical Report 的 English benchmark CoT pass@1 表：

| Model | MATH | Minerva Math | OlympiadBench | 建议 |
|---|---:|---:|---:|---|
| `Qwen2.5-Math-1.5B-Instruct` | 75.8% | 29.4% | 38.1% | 错误更多，但 CoT 质量较弱 |
| `Qwen2.5-Math-7B-Instruct` | 83.6% | 37.1% | 41.6% | 当前首选 |
| `Qwen2.5-Math-72B-Instruct` | 85.9% | 44.1% | 49.0% | 太贵，且不是第一阶段必要条件 |

因此当前选择：

```text
Qwen/Qwen2.5-Math-7B-Instruct + OlympiadBench OE_TO_maths_en_COMP
```

这个组合的预期错误率足够高，同时 7B 模型的 CoT 格式比 1.5B 更稳定。

## 3. 数据采集配置

当前配置文件：

```text
data_generation/qwen25_fhis/configs/recommended.yaml
```

核心设置：

| 字段 | 值 |
|---|---|
| dataset | `Hothan/OlympiadBench` |
| subset | `OE_TO_maths_en_COMP` |
| model | `Qwen/Qwen2.5-Math-7B-Instruct` |
| target_problems | 674 |
| samples per problem | 4 |
| temperature | 0.7 |
| top_p | 0.95 |
| max_new_tokens | 3072 |
| max_model_len | 4096 |
| completion_prefix | `Step 1:` |

生成统计：

| 指标 | 数值 |
|---|---:|
| num_traces | 2696 |
| num_problems | 674 |
| rough_correct | 1110 |
| rough_wrong | 1233 |
| rough_unknown | 353 |
| rough_accuracy | 47.38% |
| step_parse_rate | 100% |

353 条 `rough_unknown` 全部是 `final_answer_missing`，没有进入 Codex 标注或 probe 训练。

## 4. 数据产物

大文件由 `data_generation/qwen25_fhis/.gitignore` 忽略，不进入 git。

| 路径 | 内容 |
|---|---|
| `qwen25_fhis/outputs/problems.jsonl` | 674 道采样题目 |
| `qwen25_fhis/outputs/generated_traces.jsonl` | 2696 条模型推理 trace |
| `qwen25_fhis/outputs/summary.json` | 生成统计 |
| `qwen25_fhis/labels/fhis_labels.jsonl` | 本地 Codex 全量 FHIS 标注 |
| `qwen25_fhis/labels/fhis_labels_train_high.jsonl` | Codex-clean 训练标签 |
| `qwen25_fhis/features/step_hidden_states.pt` | 原始 high-confidence feature 缓存 |
| `qwen25_fhis/features/step_hidden_states_codex_clean.pt` | 当前 probe 使用的 clean feature |
| `qwen25_fhis/results/probe_metrics.json` | clean probe 指标 |
| `qwen25_fhis/results/layer_sweep.csv` | 单层 layer sweep |
| `qwen25_fhis/results/hidden_logistic_probe.joblib` | 训练后的 sklearn probe |

代码和配置：

| 路径 | 用途 |
|---|---|
| `qwen25_fhis/scripts/generate_olympiadbench_traces.py` | 生成 OlympiadBench traces |
| `qwen25_fhis/scripts/label_with_local_codex.py` | 调用本地 Codex 做 FHIS 标注 |
| `qwen25_fhis/scripts/filter_fhis_labels_for_training.py` | 生成 Codex-clean 训练标签 |
| `qwen25_fhis/scripts/filter_step_hidden_states_by_labels.py` | 从已有 hidden-state 缓存过滤 clean features |
| `qwen25_fhis/scripts/summarize_fhis_labels.py` | 汇总 Codex 标注 |
| `qwen25_fhis/schema/local_codex_label_schema.json` | Codex 输出 schema |
| `qwen25_fhis/configs/probe.yaml` | hidden-state/probe 训练配置 |

## 5. FHIS 标签

标注器：

```text
local codex exec
model = gpt-5.5
reasoning_effort = high
proxy = 127.0.0.1:7890
```

标签字段：

| 字段 | 含义 |
|---|---|
| `trace_id` | trace 唯一 ID |
| `problem_id` | 题目 ID |
| `final_correct` | Codex 判断最终答案是否正确 |
| `first_invalid_step` | 第一处关键错误 step；正确 trace 为 `null` |
| `error_type` | 错误类型 |
| `confidence` | `high` / `medium` / `low` |
| `reason` | 简短解释 |
| `rough_final_correct` | 自动答案粗扫结果，仅诊断 |

全量标注统计：

| 指标 | 数值 |
|---|---:|
| completed traces with Codex labels | 2343 |
| final_correct = true | 1130 |
| final_correct = false | 1213 |
| high confidence | 2306 |
| medium confidence | 31 |
| low confidence | 6 |

Clean 训练标签：

| 指标 | 数值 |
|---|---:|
| input labels | 2343 |
| training labels | 2128 |
| correct negative traces | 936 |
| wrong FHIS traces | 1192 |
| excluded total | 215 |
| label_not_training_usable | 203 |
| maxed_generation | 12 |

rough/Codex 冲突：

| rough | Codex | 全量 | clean train | 处理 |
|---|---:|---:|---:|---|
| False | False | 1086 | 1067 | 错误样本 |
| False | True | 147 | 123 | 以 Codex 为准，正确样本 |
| True | True | 983 | 813 | 正确样本 |
| True | False | 127 | 125 | 以 Codex 为准，错误 FHIS 样本 |

这些冲突说明 rough matcher 不能作为最终标签。`rough=False, Codex=True` 多数是等价答案没扫到；`rough=True, Codex=False` 多数是 final answer 匹配但推理过程已有 harmful invalid step。

## 6. Hidden-State Features

当前 probe 使用：

```text
data_generation/qwen25_fhis/features/step_hidden_states_codex_clean.pt
```

Feature 设置：

| 字段 | 值 |
|---|---:|
| model | `Qwen/Qwen2.5-Math-7B-Instruct` |
| layers | `[6, 13, 20, 27]` |
| hidden size / layer | 3584 |
| concatenated dim | 14336 |
| extraction point | 每个 step 结束 token |
| step rows | 8423 |
| label 0 | 7231 |
| label 1 | 1192 |

标签定义：

```text
wrong trace:
  step < FHIS  -> label 0
  step == FHIS -> label 1
  step > FHIS  -> excluded

correct trace:
  all valid steps -> label 0
```

因此 label 0 的准确含义是“不是第一处关键错误”，不等价于数学上完整正确。

`step_hidden_states_codex_clean.pt` 是从已有 `step_hidden_states.pt` 缓存中过滤得到的，没有重新跑模型。过滤依据是 `fhis_labels_train_high.jsonl`。

## 7. Probe 结果

训练配置：

```text
StandardScaler()
LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
```

训练命令：

```bash
conda run -n fhis-data-gen python -m fhis.train_probe \
  --config data_generation/qwen25_fhis/configs/probe.yaml
```

Problem-level split，避免同一道题的不同采样同时进入 train/test：

| split | problems | steps |
|---|---:|---:|
| train | 454 | 6092 |
| val | 97 | 1093 |
| test | 97 | 1238 |

测试集结果：

| 方法 | AUROC | AUPRC | recall@1 | recall@2 | top 30% budget coverage |
|---|---:|---:|---:|---:|---:|
| hidden logistic | 0.862 | 0.545 | 0.854 | 0.977 | 0.901 |
| random | 0.513 | 0.145 | 0.509 | 0.795 | 0.585 |
| step length | 0.816 | 0.462 | 0.737 | 0.930 | 0.807 |
| low mean token logprob | 0.695 | 0.233 | 0.497 | 0.789 | 0.550 |
| text tfidf logistic | 0.789 | 0.432 | 0.749 | 0.906 | 0.813 |
| final wrongness probe | 0.767 | 0.269 | 0.509 | 0.778 | 0.567 |

单层 sweep：

| layer | AUROC | AUPRC | top 30% budget coverage |
|---|---:|---:|---:|
| 6 | 0.705 | 0.310 | 0.713 |
| 13 | 0.780 | 0.405 | 0.807 |
| 20 | 0.840 | 0.510 | 0.889 |
| 27 | 0.845 | 0.524 | 0.865 |

`top 30% budget coverage = 0.901` 表示：在有错误的测试 trace 中，如果只把每条 trace 里 probe 打分最高的前 30% steps 送去验证，可以覆盖约 90.1% 的第一处关键错误。

当前 best single layer 是 layer 27，layer 20 非常接近。四层拼接仍是主配置，因为它覆盖浅层、中层和深层信号，并且主指标高于单层。

## 8. 结果解读

当前 probe 本身只是单层线性分类器，但效果好并不意外，因为它输入的不是普通文本特征，而是 Qwen 在生成每个 step 后的内部状态。四层 hidden states 拼接后是 14336 维，已经包含模型对当前上下文、推理路径、局部模式、置信度和异常状态的综合表征。线性 probe 学到的是一个方向：

```text
这个 step 的内部状态像不像 first invalid step
```

因此，这个结果说明 hidden states 中确实存在明显的 FHIS signal。它不等价于 probe 学会了数学证明，也不等价于它能跨数据集稳定判断任意数学错误。

需要谨慎解读的点：

- 当前是同一个 OlympiadBench subset 内的 problem-level split，虽然避免了同题泄漏，但仍是同分布评估。
- 表面 baseline 也不弱：step length AUROC 为 0.816，text TF-IDF AUROC 为 0.789，说明数据中存在可利用的文本和长度信号。
- hidden probe 明显更强：AUROC 0.862，AUPRC 0.545，top 30% coverage 0.901，但它的真实价值需要在 Minerva Math 或其它 holdout 上验证。
- 错误 trace 中只保留 `first_invalid_step` 及其之前的步骤，错误之后的步骤不进入 FHIS 训练目标，所以 recall@1/top-k 指标是在这个候选集合上计算的。

当前结论应表述为：

```text
Qwen2.5-Math-7B 的 step-boundary hidden states 对 FHIS 有强线性可探测信号；
当前数据和 probe pipeline 成立，但跨数据集泛化仍需验证。
```

## 9. 复现命令

推荐环境：

```bash
conda activate fhis-data-gen
```

如需代理：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

生成 traces：

```bash
python data_generation/qwen25_fhis/scripts/generate_olympiadbench_traces.py \
  --config data_generation/qwen25_fhis/configs/recommended.yaml \
  --resume
```

本地 Codex 标注，默认 `--resume` 只补缺失 trace：

```bash
python data_generation/qwen25_fhis/scripts/label_with_local_codex.py \
  --traces data_generation/qwen25_fhis/outputs/generated_traces.jsonl \
  --output data_generation/qwen25_fhis/labels/fhis_labels.jsonl \
  --schema data_generation/qwen25_fhis/schema/local_codex_label_schema.json \
  --resume
```

只重标指定 trace，建议写入 review 文件后再人工合并：

```bash
python data_generation/qwen25_fhis/scripts/label_with_local_codex.py \
  --traces data_generation/qwen25_fhis/outputs/generated_traces.jsonl \
  --output data_generation/qwen25_fhis/labels/fhis_labels_recheck.jsonl \
  --schema data_generation/qwen25_fhis/schema/local_codex_label_schema.json \
  --trace-id OE_TO_maths_en_COMP-63::sample-0
```

筛选 Codex-clean 训练标签：

```bash
python data_generation/qwen25_fhis/scripts/filter_fhis_labels_for_training.py \
  --input data_generation/qwen25_fhis/labels/fhis_labels.jsonl \
  --traces data_generation/qwen25_fhis/outputs/generated_traces.jsonl \
  --output data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl
```

从已有 hidden-state 缓存过滤 clean feature，不重新跑模型：

```bash
python data_generation/qwen25_fhis/scripts/filter_step_hidden_states_by_labels.py \
  --input data_generation/qwen25_fhis/features/step_hidden_states.pt \
  --labels data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl \
  --output data_generation/qwen25_fhis/features/step_hidden_states_codex_clean.pt
```

如果必须从头提取 hidden states：

```bash
python -m fhis.extract_hidden_states_transformers \
  --config data_generation/qwen25_fhis/configs/probe.yaml
```

训练 probe：

```bash
python -m fhis.train_probe \
  --config data_generation/qwen25_fhis/configs/probe.yaml
```

查看 summary：

```bash
python data_generation/qwen25_fhis/scripts/summarize_fhis_labels.py \
  --labels data_generation/qwen25_fhis/labels/fhis_labels.jsonl
```

## 10. 质量边界

当前数据已经可以直接训练第一版 probe，但还不是最终论文级数据：

- Codex FHIS 是自动标注，关键样本仍建议抽查。
- 未完成且没有 final answer 的 353 条 trace 没有进入训练。
- 打满 `max_new_tokens` 的 12 条 high-confidence runaway trace 已从 clean 训练集排除。
- 正确 trace 被用作 negative；如果正确 trace 中存在无害绕路或可修复错误，当前 schema 不细分。
- hidden states 来自同一个 frozen Qwen2.5 checkpoint；换模型后需要重新提取 hidden states，并通常重新训练 probe。
- 当前评估仍是同一 OlympiadBench subset 内的 problem split，下一步需要跨数据集 holdout。

## 11. 下一步

优先级：

1. 抽查 rough/Codex 冲突样本，尤其 `rough=True, Codex=False`。
2. 加 Minerva Math holdout，测试跨数据集泛化。
3. 保留当前 Qwen2.5 probe 作为 baseline，再考虑 Qwen3 或其它模型。
4. 如果要接 Lean，只对 probe 排名前 30% 的候选 step 先做验证，以控制预算。

扩容时仍必须按 `problem_id` split，不能按 step 或 trace 随机 split。同一道题的不同采样不能同时出现在 train 和 test。

## 12. 参考

- Qwen2.5-Math Technical Report: https://arxiv.org/html/2409.12122
- Qwen2.5-Math official blog: https://qwenlm.github.io/blog/qwen2.5-math/
- OlympiadBench dataset card: https://huggingface.co/datasets/Hothan/OlympiadBench
