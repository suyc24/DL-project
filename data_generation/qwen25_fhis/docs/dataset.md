# Qwen2.5 FHIS Probe 数据集说明

本文档记录当前仓库中已经构建完成的第一版 FHIS probe 训练数据。目标是让后续实验可以直接从 trace、Codex 标注和 hidden state features 进入 probe 训练，而不是重新设计采集流程。

## 1. 数据集目标

这个数据集用于训练 step-level probe：

```text
输入：Qwen2.5-Math-7B-Instruct 在每个推理 step 结束位置的 hidden states
输出：这个 step 是否是第一处关键错误，或是否值得优先送去形式化/外部验证
```

第一版重点不是追求最高数学正确率，而是构造足够多、格式稳定、可定位第一处错误的推理轨迹。模型正确率不能太高，否则 wrong trace 不够，probe 会缺少正例。

## 2. 数据来源

采集设置：

| 字段 | 值 |
|---|---|
| 题库 | `Hothan/OlympiadBench` |
| subset | `OE_TO_maths_en_COMP` |
| 题型 | 英文、text-only、open-ended、competition math |
| 生成模型 | `Qwen/Qwen2.5-Math-7B-Instruct` |
| 题目数 | 200 |
| 每题采样数 | 4 |
| 总 trace 数 | 800 |
| temperature | 0.7 |
| top_p | 0.95 |
| max_new_tokens | 3072 |
| completion_prefix | `Step 1:` |

官方正确率参考来自 Qwen2.5-Math technical report：`Qwen2.5-Math-7B-Instruct` 在 OlympiadBench 的 CoT pass@1 约为 41.6%。这意味着它会保留较高比例的错误推理，适合第一阶段 probe 数据构造。

## 3. 当前产物

生成和训练产物都收拢在 `data_generation/qwen25_fhis/` 下。大文件已经被 bundle 内的 `.gitignore` 排除，不进入 git commit。

| 路径 | 内容 | 当前规模 |
|---|---|---:|
| `data_generation/qwen25_fhis/outputs/problems.jsonl` | 采样到的 200 道原题 | ignored |
| `data_generation/qwen25_fhis/outputs/generated_traces.jsonl` | 800 条模型推理 trace | 约 21 MB |
| `data_generation/qwen25_fhis/outputs/summary.json` | 生成统计 | ignored |
| `data_generation/qwen25_fhis/labels/fhis_labels.jsonl` | 本地 Codex FHIS 全量标注 | 约 848 KB |
| `data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl` | 高置信度训练标签 | ignored |
| `data_generation/qwen25_fhis/features/step_hidden_states.pt` | step hidden state features | 约 136 MB |
| `data_generation/qwen25_fhis/results/probe_metrics.json` | probe 训练评估结果 | ignored |
| `data_generation/qwen25_fhis/results/layer_sweep.csv` | 单层 hidden state probe sweep | ignored |

代码、配置和 schema 会进入 git：

| 路径 | 用途 |
|---|---|
| `data_generation/qwen25_fhis/scripts/generate_olympiadbench_traces.py` | 生成 OlympiadBench traces |
| `data_generation/qwen25_fhis/scripts/label_with_local_codex.py` | 调用本地 Codex 做 FHIS 标注 |
| `data_generation/qwen25_fhis/schema/local_codex_label_schema.json` | Codex 标注 JSON schema |
| `data_generation/qwen25_fhis/scripts/filter_fhis_labels_for_training.py` | 筛选可训练的高置信度标签 |
| `data_generation/qwen25_fhis/scripts/summarize_fhis_labels.py` | 汇总标注分布 |
| `data_generation/qwen25_fhis/configs/recommended.yaml` | trace 生成配置 |
| `data_generation/qwen25_fhis/configs/probe.yaml` | hidden state 和 probe 配置 |

当前 probe v0 的单独报告见 `data_generation/qwen25_fhis/docs/probe_v0.md`。

## 4. 生成统计

`data_generation/qwen25_fhis/outputs/summary.json` 当前结果：

| 指标 | 数值 |
|---|---:|
| num_traces | 800 |
| num_problems | 200 |
| rough_correct | 326 |
| rough_wrong | 376 |
| rough_unknown | 98 |
| rough_accuracy | 46.44% |
| step_parse_rate | 100% |

`rough_unknown` 主要是未完成或无法可靠抽取最终答案的 trace，不作为第一版高质量训练标签使用。

## 5. Codex FHIS 标注

标注器：

```text
本地 codex exec
model = gpt-5.5
reasoning_effort = high
proxy = 127.0.0.1:7890
```

标注目标是找出每条 trace 的 first harmful incorrect step，简称 FHIS。每条标注包含：

| 字段 | 含义 |
|---|---|
| `trace_id` | trace 唯一 ID |
| `final_correct` | Codex 判断最终答案是否正确 |
| `first_error_step` | 第一处关键错误 step；正确 trace 为 `null` |
| `error_type` | 错误类型 |
| `confidence` | 标注置信度 |
| `train_usable` | 是否建议用于 probe 训练 |
| `rationale` | 简短解释 |

当前标注统计：

| 指标 | 数值 |
|---|---:|
| 已标注 completed traces | 702 |
| final_correct = true | 354 |
| final_correct = false | 348 |
| high confidence | 697 |
| medium confidence | 4 |
| low confidence | 1 |
| 高置信度训练样本 | 646 |
| 正确 negative traces | 300 |
| 错误 FHIS traces | 346 |
| 排除样本 | 56 |

排除逻辑包括：低置信度、未完成、最终答案和 trace 结构冲突、或 Codex 认为不适合直接训练 probe 的样本。

## 6. Hidden State Features

hidden states 已保存：

```text
data_generation/qwen25_fhis/features/step_hidden_states.pt
```

提取设置：

| 字段 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-Math-7B-Instruct` |
| layers | `[6, 13, 20, 27]` |
| hidden size | 3584 |
| 拼接后 feature dim | 14336 |
| step rows | 2444 |
| label 0 | 2098 |
| label 1 | 346 |

每个 step 的 feature 是在该 step 结束位置取选定层 hidden states 后拼接得到。第一版没有保存全 token hidden states，因为 probe 训练只需要 step boundary 表征，保存全 token 会让数据体积显著增大。

## 7. Probe 结果

当前使用 `fhis.train_probe` 训练 logistic probe，按 problem split，避免同一道题的不同采样同时进入 train/test。

split：

| split | problems | steps |
|---|---:|---:|
| train | 135 | 1662 |
| val | 29 | 414 |
| test | 29 | 368 |

测试集结果：

| 方法 | AUROC | AUPRC | top 30% budget coverage |
|---|---:|---:|---:|
| hidden logistic | 0.873 | 0.585 | 0.851 |
| text tfidf logistic | 0.766 | 0.403 | 0.766 |
| low mean token logprob | 0.757 | 0.252 | 0.681 |
| step length | 0.794 | 0.388 | 0.723 |
| random | 0.551 | 0.147 | 0.596 |

单层 sweep：

| layer | AUROC | AUPRC | top 30% budget coverage |
|---|---:|---:|---:|
| 6 | 0.770 | 0.335 | 0.830 |
| 13 | 0.853 | 0.445 | 0.851 |
| 20 | 0.886 | 0.598 | 0.872 |
| 27 | 0.862 | 0.579 | 0.872 |

当前 best single layer 是 layer 20。四层拼接仍然是主配置，因为它更稳定地覆盖浅层、中层和深层信号。

## 8. 复现命令

推荐在 conda 环境 `fhis-data-gen` 中运行。若需要走代理：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

生成 traces：

```bash
conda run -n fhis-data-gen python data_generation/qwen25_fhis/scripts/generate_olympiadbench_traces.py \
  --config data_generation/qwen25_fhis/configs/recommended.yaml \
  --resume
```

本地 Codex 标注：

```bash
conda run -n fhis-data-gen python data_generation/qwen25_fhis/scripts/label_with_local_codex.py \
  --traces data_generation/qwen25_fhis/outputs/generated_traces.jsonl \
  --output data_generation/qwen25_fhis/labels/fhis_labels.jsonl \
  --schema data_generation/qwen25_fhis/schema/local_codex_label_schema.json \
  --resume
```

筛选训练标签：

```bash
conda run -n fhis-data-gen python data_generation/qwen25_fhis/scripts/filter_fhis_labels_for_training.py \
  --input data_generation/qwen25_fhis/labels/fhis_labels.jsonl \
  --output data_generation/qwen25_fhis/labels/fhis_labels_train_high.jsonl
```

提取 hidden states：

```bash
conda run -n fhis-data-gen python -m fhis.extract_hidden_states_transformers \
  --config data_generation/qwen25_fhis/configs/probe.yaml
```

训练 probe：

```bash
conda run -n fhis-data-gen python -m fhis.train_probe \
  --config data_generation/qwen25_fhis/configs/probe.yaml
```

查看标注摘要：

```bash
conda run -n fhis-data-gen python data_generation/qwen25_fhis/scripts/summarize_fhis_labels.py \
  --labels data_generation/qwen25_fhis/labels/fhis_labels.jsonl
```

## 9. 质量边界

当前数据集已经可以直接用于第一版 probe 训练，但还不是最终论文级数据：

- Codex FHIS 是自动标注，关键样本仍建议人工抽查。
- `rough_unknown` 未完成 trace 已被排除，不应混入训练。
- 正确 trace 被用作 negative；如果正确 trace 中存在无害绕路或可修复错误，当前 schema 不会细分。
- hidden state 来自同一个 frozen Qwen2.5 checkpoint；换模型后需要重新提取 hidden states，并通常重新训练 probe。
- 当前 feature 是 step boundary 表征，不包含全 token trajectory。

第一版结论是：数据规模、错误比例、step 解析率、FHIS 标注和 hidden state features 都已经满足直接训练 probe 的最低要求。
