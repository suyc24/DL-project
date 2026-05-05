# AG-SFV 项目说明与第一阶段数据采集建议

本文档用于远程集群上第一轮实验启动前的统一说明。目标不是一次性完成完整 Lean/FHIS pipeline，而是先构建一个适合训练 probe model 的错误 CoT 数据集。

## 1. 项目核心目标

项目名称：Activation-Gated Selective Formal Verification，简称 AG-SFV。

核心问题是：大模型在数学推理中会生成看似合理的 chain-of-thought，但中间某一步可能已经出错。完整地把每一步都送入 Lean 做形式化验证很贵，所以我们希望训练一个轻量 probe，从 LLM 的 hidden states 判断“哪一步最值得验证”。

最终系统可以拆成四层：

1. 生成结构化数学推理：让冻结 LLM 输出 `Step 1`, `Step 2`, ..., `Final Answer`。
2. 提取 step-boundary hidden states：在每个步骤结束 token 处取若干层 hidden states。
3. 构造标签：第一阶段先用最终答案对错构造 weak labels；后续再升级成 FHIS 标签。
4. 训练 probe：输入 hidden states，预测 trace 是否会错，或预测当前 step 是否是高价值验证点。

第一阶段的关键不是 Lean，而是确认 hidden states 是否包含“推理即将失败/已经不可靠”的信号。

## 2. 为什么第一步先做数据采集

probe model 需要足够多的错误 CoT 路径。如果题太简单，模型最终答案正确率很高，错误样本太少；如果题太难，CoT 会变成乱猜或格式崩坏，也不利于学习“局部推理错误”。

第一批数据的理想区间：

- 最终答案错误率：约 40% 到 70%。
- CoT 格式稳定：能够解析出多个 step。
- 答案可自动或半自动比对。
- 题目不依赖图片，避免视觉信息和 OCR 干扰。

因此，MATH Level 5 不适合作为主数据集。官方 Qwen2.5-Math 报告中，`Qwen2.5-Math-7B-Instruct` 在 MATH 上 CoT pass@1 为 83.6%，错误率只有约 16.4%；但在 Minerva Math 和 OlympiadBench 上分别为 37.1% 和 41.6%，更适合构建错误 CoT 数据。

## 3. 推荐模型

首选：

```text
Qwen/Qwen2.5-Math-7B-Instruct
```

理由：

- 7B 在 A800-80G 上运行很稳，A10/3090 也可以跑生成。
- CoT 质量明显好于 1.5B，格式更稳定。
- 在 OlympiadBench / Minerva Math 上错误率足够高，不会像 MATH 那样过于简单。

备选：

```text
Qwen/Qwen2.5-Math-1.5B-Instruct
```

使用场景：

- 如果 7B 在选定题集上实际错误率低于 40%，可以换 1.5B。
- 但 1.5B 更容易出现格式混乱、无效推理，标签噪声可能更大。

暂不推荐：

```text
Qwen2.5-Math-14B/72B
Qwen3 thinking 系列
```

原因是第一阶段不是追求最高正确率，而是要稳定地产生可学习的错误路径。

## 4. 推荐题目集

第一优先级：

```text
OlympiadBench: OE_TO_maths_en_COMP
```

含义：

- `OE`：open-ended，非选择题。
- `TO`：text-only，不依赖图片。
- `maths`：数学题，不混入物理。
- `en`：英文。
- `COMP`：competition 难度。

Hugging Face 数据集卡显示该 subset 有 674 条，字段包括 `question`, `solution`, `final_answer`, `question_type`, `subject`, `language` 等。这个规模非常适合第一轮 pilot。

第二优先级：

```text
Minerva Math
```

推荐作为第二批或混合评估集。它比 MATH 难很多，Qwen2.5-Math-7B-Instruct 的 CoT pass@1 约 37.1%。

控制组：

```text
MATH Level 5
```

只建议作为 easy/control subset，不建议作为主训练数据。

暂时避免：

```text
OlympiadBench multimodal subsets
Geometry image-heavy problems
Physics subsets
Multiple-choice subsets
```

这些会引入图片、单位、选择题技巧或视觉信息，第一阶段不利于稳定提取 CoT hidden-state signal。

## 5. 官方正确率参考

来自 Qwen2.5-Math Technical Report 的 English benchmark CoT pass@1 表：

| Model | MATH | Minerva Math | OlympiadBench | 建议 |
|---|---:|---:|---:|---|
| Qwen2.5-Math-1.5B-Instruct | 75.8% | 29.4% | 38.1% | 错误更多，但 CoT 质量较弱 |
| Qwen2.5-Math-7B-Instruct | 83.6% | 37.1% | 41.6% | 第一阶段首选 |
| Qwen2.5-Math-72B-Instruct | 85.9% | 44.1% | 49.0% | 太贵，且错误率不一定更合适 |

粗略对应错误率：

| Model + Dataset | 预期错误率 |
|---|---:|
| 7B + MATH | 约 16.4% |
| 7B + Minerva Math | 约 62.9% |
| 7B + OlympiadBench | 约 58.4% |
| 1.5B + OlympiadBench | 约 61.9% |

结论：第一轮应主跑 `Qwen2.5-Math-7B-Instruct + OlympiadBench text-only open-ended competition`。

## 6. 第一轮数据采集配置

推荐配置：

```yaml
model:
  name: Qwen/Qwen2.5-Math-7B-Instruct
  tensor_parallel_size: 1
  dtype: auto
  max_model_len: 8192

generation:
  temperature: 0.7
  top_p: 0.95
  n_samples_per_problem: 4
  max_new_tokens: 4096
  logprobs: 5

dataset:
  name: Hothan/OlympiadBench
  subset: OE_TO_maths_en_COMP
  target_problems: 200
```

第一批规模：

```text
Problems: 200
Samples per problem: 4
Total traces: 800
Expected wrong traces: 约 450 左右
```

如果第一批实际错误率低于 40%，下一轮改：

```text
Qwen/Qwen2.5-Math-1.5B-Instruct
```

如果实际错误率高于 75%，或者 step 格式明显崩坏，回到 7B 并混入少量 MATH Level 5 / Minerva 中等难度题。

## 7. 每条 trace 建议保存字段

每条生成结果保存为 JSONL：

```json
{
  "trace_id": "...",
  "problem_id": "...",
  "dataset": "OlympiadBench",
  "subset": "OE_TO_maths_en_COMP",
  "problem": "...",
  "reference_answer": ["..."],
  "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
  "sample_index": 0,
  "generation_config": {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 4096
  },
  "completion": "...",
  "steps": [
    {"index": 1, "text": "...", "start_char": 0, "end_char": 120}
  ],
  "final_answer": "...",
  "rough_final_correct": false,
  "token_ids": [],
  "token_logprobs": []
}
```

后续 hidden-state features 单独保存，避免 JSONL 过大：

```text
data/generated_traces.jsonl
data/step_hidden_states.pt
```

## 8. 第一阶段 probe 标签

第一阶段先使用 outcome weak label：

- trace 最终答案正确：该 trace 的 step hidden states 暂标为 negative。
- trace 最终答案错误：该 trace 的 step hidden states 暂标为 positive 或用于 trace-level wrongness probe。

这不是最终 FHIS 标签，但足够回答第一个问题：

```text
hidden states 能不能预测一条 CoT 最终会错？
```

如果该实验中 hidden-state probe 明显强于 random、token logprob、text TF-IDF baseline，再进入第二阶段：

```text
OpenAI/人工/Lean 辅助标注 first_invalid_step，构造 FHIS 标签。
```

## 9. 第一阶段评价指标

至少报告：

- AUROC：区分正确/错误 trace 的能力。
- AUPRC：错误 trace 检出能力。
- top-k error coverage：如果只检查 top 10%/20% 高风险 step，能覆盖多少错误 trace。
- baseline comparison：
  - random
  - low token logprob
  - step index
  - step length
  - text TF-IDF logistic

第一轮成功标准：

```text
hidden-state probe 在 AUROC 和 top-20% error coverage 上明显优于 random/logprob baseline。
```

## 10. 集群建议

生成 traces：

```text
GPU: A10 / RTX3090 / A800 单卡均可
推荐: A800-80G x 1，省心
```

抽 hidden states：

```text
推荐: A800-80G x 1
原因: 长上下文 + 多层 hidden states 更吃显存和内存
```

首轮不要多卡，不需要 A800 多卡。

## 11. 参考来源

- Qwen2.5-Math Technical Report: https://arxiv.org/html/2409.12122
- Qwen2.5-Math official blog: https://qwenlm.github.io/blog/qwen2.5-math/
- OlympiadBench dataset card: https://huggingface.co/datasets/Hothan/OlympiadBench
- Project repository: https://github.com/suyc24/DL-project
