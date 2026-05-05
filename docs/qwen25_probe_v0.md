# Qwen2.5 FHIS Probe v0

本文档记录当前已经训练完成的第一版 FHIS probe。它用于从 Qwen2.5-Math-7B-Instruct 的 step-boundary hidden states 中预测第一处关键错误 step。

## 1. 训练数据

| 字段 | 值 |
|---|---:|
| 生成模型 | `Qwen/Qwen2.5-Math-7B-Instruct` |
| 数据集 | `Hothan/OlympiadBench`, `OE_TO_maths_en_COMP` |
| problems | 200 |
| generated traces | 800 |
| completed labeled traces | 702 |
| high-confidence training traces | 646 |
| wrong FHIS traces | 346 |
| correct negative traces | 300 |
| step rows | 2444 |
| positive FHIS rows | 346 |
| negative rows | 2098 |

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

## 2. Feature

hidden state 文件：

```text
data_generation/features/step_hidden_states.pt
```

feature 设置：

| 字段 | 值 |
|---|---:|
| layers | `[6, 13, 20, 27]` |
| hidden size / layer | 3584 |
| concatenated dim | 14336 |
| extraction point | 每个 step 结束 token |

当前没有保存全 token hidden states。probe v0 只使用 step boundary 的多层拼接表征。

## 3. 模型

probe v0 是一个 sklearn logistic regression pipeline：

```text
StandardScaler()
LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
```

配置文件：

```text
data_generation/qwen25_probe_config.yaml
```

训练命令：

```bash
conda run -n fhis-data-gen python -m fhis.train_probe \
  --config data_generation/qwen25_probe_config.yaml
```

训练后本地保存：

```text
data_generation/results/hidden_logistic_probe.joblib
```

这个文件在 `data_generation/results/` 下，默认不进入 git。git 中提交的是训练代码、配置和文档。

## 4. 评估结果

problem-level split：

| split | problems | steps |
|---|---:|---:|
| train | 135 | 1662 |
| val | 29 | 414 |
| test | 29 | 368 |

测试集结果：

| 方法 | AUROC | AUPRC | recall@1 | recall@2 | top 30% budget coverage |
|---|---:|---:|---:|---:|---:|
| hidden logistic | 0.873 | 0.585 | 0.851 | 0.957 | 0.851 |
| random | 0.551 | 0.147 | 0.574 | 0.809 | 0.596 |
| step length | 0.794 | 0.388 | 0.702 | 0.915 | 0.723 |
| low mean token logprob | 0.757 | 0.252 | 0.596 | 0.872 | 0.681 |
| text tfidf logistic | 0.766 | 0.403 | 0.702 | 0.894 | 0.766 |
| final wrongness probe | 0.764 | 0.279 | 0.447 | 0.745 | 0.511 |

单层 sweep：

| layer | AUROC | AUPRC | top 30% budget coverage |
|---|---:|---:|---:|
| 6 | 0.770 | 0.335 | 0.830 |
| 13 | 0.853 | 0.445 | 0.851 |
| 20 | 0.886 | 0.598 | 0.872 |
| 27 | 0.862 | 0.579 | 0.872 |

当前 best single layer 是 layer 20。多层拼接 probe 的主结果已经显著强于 random、token logprob 和 text baseline。

## 5. 结果解释

`top 30% budget coverage = 0.851` 表示：在有错误的测试 trace 中，如果只把每条 trace 里 probe 打分最高的前 30% steps 送去验证，可以覆盖约 85.1% 的第一处关键错误。

这说明当前 hidden state signal 足够支持第一版 selective verification gate。但测试集中只有 47 条 wrong traces，因此它更适合作为可行性结果，不应被当成最终泛化结论。

## 6. 下一步扩充计划

优先扩同分布题目数，而不是先增加同题采样数：

```text
当前：200 problems x 4 samples = 800 traces
下一步：600 problems x 4 samples = 2400 traces
上限：674 problems x 4 samples = 2696 traces
```

预期目标：

| 阶段 | 目标 |
|---|---|
| OlympiadBench 600 题 | 约 900-1100 条 high-confidence wrong FHIS traces |
| OlympiadBench 全量 | 至少 1000 条 high-confidence wrong FHIS traces |
| Minerva Math holdout | 测跨数据集泛化 |

扩容时仍然必须按 `problem_id` split，不能按 step 或 trace 随机 split。同一道题的不同采样不能同时出现在 train 和 test。

## 7. 当前结论

probe v0 已经满足 pipeline 验证条件：

- hidden state probe 明显强于非激活 baseline。
- FHIS 标签、hidden state features、训练脚本和评估指标已经闭环。
- 当前数据量偏小，需要扩容后再判断泛化稳定性。
