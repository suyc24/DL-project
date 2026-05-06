# Qwen2.5 FHIS Probe v0

本文档记录当前已经训练完成的第一版 FHIS probe。它用于从 Qwen2.5-Math-7B-Instruct 的 step-boundary hidden states 中预测第一处关键错误 step。

## 1. 训练数据

| 字段 | 值 |
|---|---:|
| 生成模型 | `Qwen/Qwen2.5-Math-7B-Instruct` |
| 数据集 | `Hothan/OlympiadBench`, `OE_TO_maths_en_COMP` |
| problems | 674 |
| generated traces | 2696 |
| completed labeled traces | 2343 |
| high-confidence training traces | 2140 |
| wrong FHIS traces | 1203 |
| correct negative traces | 937 |
| step rows | 9438 |
| positive FHIS rows | 1203 |
| negative rows | 8235 |

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
data_generation/qwen25_fhis/features/step_hidden_states.pt
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
data_generation/qwen25_fhis/configs/probe.yaml
```

训练命令：

```bash
conda run -n fhis-data-gen python -m fhis.train_probe \
  --config data_generation/qwen25_fhis/configs/probe.yaml
```

训练后本地保存：

```text
data_generation/qwen25_fhis/results/hidden_logistic_probe.joblib
```

这个文件在 `data_generation/qwen25_fhis/results/` 下，默认不进入 git。git 中提交的是训练代码、配置和文档。

## 4. 评估结果

problem-level split：

| split | problems | steps |
|---|---:|---:|
| train | 461 | 6641 |
| val | 99 | 1400 |
| test | 98 | 1397 |

测试集结果：

| 方法 | AUROC | AUPRC | recall@1 | recall@2 | top 30% budget coverage |
|---|---:|---:|---:|---:|---:|
| hidden logistic | 0.790 | 0.370 | 0.766 | 0.880 | 0.813 |
| random | 0.440 | 0.124 | 0.464 | 0.724 | 0.547 |
| step length | 0.755 | 0.360 | 0.698 | 0.917 | 0.792 |
| low mean token logprob | 0.670 | 0.209 | 0.490 | 0.766 | 0.557 |
| text tfidf logistic | 0.746 | 0.361 | 0.682 | 0.870 | 0.792 |
| final wrongness probe | 0.673 | 0.225 | 0.469 | 0.776 | 0.536 |

单层 sweep：

| layer | AUROC | AUPRC | top 30% budget coverage |
|---|---:|---:|---:|
| 6 | 0.670 | 0.231 | 0.740 |
| 13 | 0.742 | 0.340 | 0.708 |
| 20 | 0.773 | 0.354 | 0.771 |
| 27 | 0.753 | 0.335 | 0.766 |

当前 best single layer 是 layer 20。多层拼接 probe 的主结果已经显著强于 random、token logprob 和 text baseline。

## 5. 结果解释

`top 30% budget coverage = 0.813` 表示：在有错误的测试 trace 中，如果只把每条 trace 里 probe 打分最高的前 30% steps 送去验证，可以覆盖约 81.3% 的第一处关键错误。

这说明当前 hidden state signal 仍然显著强于 random 和 token-logprob baseline。测试集中有 192 条 wrong traces，比 200 题版本稳定得多；但它仍是同一 OlympiadBench subset 内的评估，跨数据集泛化还需要 Minerva Math 等 holdout。

## 6. 下一步扩充计划

优先扩同分布题目数，而不是先增加同题采样数：

```text
当前：674 problems x 4 samples = 2696 traces
已完成：全量 OE_TO_maths_en_COMP subset
下一步：添加 Minerva Math holdout 或跨 subset 评估
```

预期目标：

| 阶段 | 目标 |
|---|---|
| OlympiadBench 全量 | 已完成，1203 条 high-confidence wrong FHIS traces |
| Minerva Math holdout | 下一步测跨数据集泛化 |
| Qwen layer ablation | 对比 layer 20 单层和四层拼接 |

扩容时仍然必须按 `problem_id` split，不能按 step 或 trace 随机 split。同一道题的不同采样不能同时出现在 train 和 test。

## 7. 当前结论

probe v0 已经满足 pipeline 验证条件：

- hidden state probe 明显强于非激活 baseline。
- FHIS 标签、hidden state features、训练脚本和评估指标已经闭环。
- 当前同分布数据量已经足够，下一步重点是跨数据集泛化和标注抽查。
