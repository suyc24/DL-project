# FHIS Probe Classifier v2 训练计划

本文档基于仓库 README、`data_generation/README.md`、当前 probe 配置与结果文件整理，目标是把第一版 `data_generation/qwen25_fhis/results/hidden_mlp_probe.joblib` 升级为更适合 test-time 干预的 classifier。

## 1. 当前任务和基线

仓库主目标是 Activation-Gated Selective Formal Verification, AG-SFV：在模型生成结构化数学推理时，读取每个 `Step k` 结束位置的 hidden states，判断当前 step 是否为 first harmful invalid step, FHIS，从而决定是否把该 step 送去验证或触发重试。

当前主数据和模型：

| 项目 | 当前设置 |
|---|---:|
| 生成模型 | `Qwen/Qwen2.5-Math-7B-Instruct` |
| 数据源 | OlympiadBench `OE_TO_maths_en_COMP` |
| 题目数 | 674 |
| generated traces | 2696 |
| Codex-clean training traces | 2128 |
| wrong FHIS traces | 1192 |
| correct negative traces | 936 |
| step rows | 8423 |
| positive FHIS rows | 1192 |
| negative rows | 7231 |
| hidden feature | layers `[6, 13, 20, 27]` concat, dim 14336 |

当前 `probe_metrics.json` 中的主要结果：

| 模型 | AUROC | AUPRC | recall@1 | recall@2 | top 30% coverage |
|---|---:|---:|---:|---:|---:|
| hidden MLP | 0.874 | 0.521 | 0.860 | 0.947 | 0.895 |
| hidden logistic | 0.862 | 0.545 | 0.854 | 0.977 | 0.901 |
| step length baseline | 0.816 | 0.462 | 0.737 | 0.930 | 0.807 |
| text TF-IDF logistic | 0.789 | 0.432 | 0.749 | 0.906 | 0.813 |

这个结果已经证明 hidden states 内有 FHIS signal，但还不够支撑直接 test-time 干预。关键问题不是 AUROC 不够高，而是 thresholded online action 的代价很高：为了获得高 FHIS recall，正确 trace 和 FHIS 之前的正常 steps 会被过早误触发。

例如 `online_fhis_recall_constrained_eval.json` 显示，在 OlympiadBench test split 上 hidden MLP 若要求 FHIS step recall 0.90，observable non-FHIS step FPR 约 0.331，correct-trace false stop rate 约 0.811，pre-FHIS stop rate 约 0.415。若要求 recall 0.95，non-FHIS FPR 约 0.482，correct-trace false stop rate 约 0.895。也就是说，当前 classifier 更像一个不错的 trace 内排序器，还不是一个足够稳定的 online hazard detector。

## 2. v2 目标

v2 不应只优化 row-level AUROC/AUPRC，而应直接优化 online 干预所需的风险曲线。

建议主目标：

| 指标 | v1 当前状态 | v2 目标 |
|---|---:|---:|
| in-domain AUPRC | 0.52-0.55 | >= 0.65 |
| in-domain top 30% FHIS coverage | ~0.90 | >= 0.95 |
| FHIS recall 0.90 时 step FPR | ~0.33 | <= 0.15 |
| FHIS recall 0.90 时 correct-trace false stop | ~0.81 | <= 0.35 |
| FHIS recall 0.90 时 pre-FHIS stop on wrong traces | ~0.42 | <= 0.20 |
| MATH Level 5 holdout wrong traces | 22 | >= 500 |

如果 probe 只用于 routing external verifier，较高 false positive 可以接受一些；如果 probe 自己触发 step retry，false positive 必须显著下降。因此训练和评估需要区分两种模式：

| 使用模式 | classifier 目标 |
|---|---|
| route-to-verifier | 高 recall，高 top-k coverage，控制验证预算 |
| probe-only retry | 高 precision / 低 early false stop，校准后的可行动阈值 |

## 3. 数据集选择

当前数据只来自 OlympiadBench 的一个 subset，同分布表现不能代表 online 泛化。v2 数据应覆盖题型、难度、生成分布和 retry 分布。

### 3.1 保留当前数据作为 in-domain baseline

继续保留 `OlympiadBench OE_TO_maths_en_COMP`，但只把它作为 in-domain split 和回归测试。当前 subset 的优势是错误率合适、step 格式稳定、已有完整生成和标注；劣势是题型和文本分布较窄，step length/text baseline 已经很强，说明存在数据捷径。

### 3.2 扩展到多题型数学数据

优先加入：

| 数据源 | 用途 |
|---|---|
| OlympiadBench 其它英文数学 subset | 扩展同类竞赛题分布，降低单 subset shortcut |
| MATH Level 4-5 / Hendrycks Math | 当前已有小 holdout，应扩成正式 cross-dataset split |
| Minerva-style math | 检查长链、多公式题上的泛化 |
| AIME/AMC 风格题 | 测试短题干但高推理密度场景 |

选题原则：

- 优先选择 Qwen2.5-Math-7B pass rate 大约 30%-70% 的题集。太简单会缺少 positive FHIS，太难会产生格式崩坏或无意义错误。
- 每个 problem 采样多条 trace，但 split 必须 problem-disjoint。同一题的不同采样不能跨 train/val/test。
- 保留一个完全 dataset-disjoint test，例如 MATH Level 5，不参与调参。
- 题型维度至少覆盖 algebra、number theory、geometry、combinatorics、precalculus。

### 3.3 加入 online 分布数据

当前 classifier 训练在 offline full trace 的 step boundary 上；online retry 时，模型会看到“前一步被指出可疑，请重做 Step k”的反馈 prompt，这会造成分布偏移。

必须新增三类 online 数据：

| 数据类型 | 标注方式 | 目的 |
|---|---|---|
| 首次生成 step | 当前 FHIS 标注 | 保持与现有数据兼容 |
| 被 probe flagged 后重写的 step | 标注当前 step 是否仍 harmful invalid | 训练 retry 后分布 |
| probe 高分但实际正确的 step | 强制加入 hard negative | 降低 false stop |

建议从 `probe_retry_router` 和 `online_router` 日志中采样被当前 MLP 高分误杀的 examples，做 hard-negative mining。当前最大瓶颈正是 false positives，所以这类数据比随机增加 easy negatives 更有价值。

## 4. 标注方案

### 4.1 当前标签的优点和缺口

当前 Codex FHIS schema 已包含：

| 字段 | 含义 |
|---|---|
| `final_correct` | 最终答案是否正确 |
| `first_invalid_step` | 第一处 harmful invalid step |
| `error_type` | 错误类型 |
| `confidence` | high / medium / low |
| `reason` | 简短解释 |

这个 schema 足以训练 v1 binary probe，但 v2 需要更细的 online supervision。

主要缺口：

- 当前 clean filter 对 `final_correct=true` 且 `first_invalid_step != null` 的 recovered trace 会排除；如果目标是发现中途 harmful invalid step，这类样本应保留为 positive 或单独的 `recovered_invalid` 类。
- post-FHIS steps 被排除是合理的，因为它们不适合作为“第一处错误”正例；但它们可以作为 auxiliary `after_invalid` 类，不应混成 negative。
- `error_type` 目前自由文本过散，需要规范化，否则无法稳定做多任务训练。
- 对 hard negatives 缺少专门标注，例如“很长但正确”“跳步但无害”“符号不严谨但数学正确”。

### 4.2 v2 标签 schema

建议升级为 trace-level + step-level 混合 schema：

```json
{
  "trace_id": "...",
  "final_correct": true,
  "first_harmful_invalid_step": 4,
  "confidence": "high",
  "adjudication": "single|consensus|human_reviewed",
  "steps": [
    {
      "step_index": 1,
      "validity": "valid|benign_gap|harmful_invalid|after_invalid|ambiguous",
      "is_first_harmful_invalid": false,
      "error_type": null,
      "repairability": "easy|hard|unknown",
      "reason": "..."
    }
  ]
}
```

主 binary label 仍然是 `is_first_harmful_invalid`，但训练可以额外使用 `validity`、`error_type`、`repairability` 做辅助任务。

### 4.3 标注质量控制

建议采用三层标注：

| 层级 | 范围 | 做法 |
|---|---|---|
| 自动单标 | 全量数据 | 使用当前 Codex prompt 的增强版，强制 step-level 输出 |
| 双模型/双 prompt 复标 | 20%-30% 样本、全部 rough/Codex 冲突、全部 high-score FP/FN | 两个独立判断，不一致则进入 adjudication |
| 人工或高强度 adjudication | gold dev/test、边界样本、线上误杀样本 | 形成小而可靠的 calibration/eval set |

必须重点复查：

- 当前 248 条 clean 中的 rough/Codex final-correct 冲突。
- 当前 classifier 在 val/test 中高分 false positives。
- 当前 classifier missed 的 FHIS false negatives。
- `first_invalid_step=1` 的样本。当前 1192 个 wrong traces 中有 242 个 FHIS 在 Step 1，容易和“题目理解/开局套路”混淆。

## 5. 模型和 feature 选择

### 5.1 生成模型选择

如果 test-time 干预目标仍是 `Qwen/Qwen2.5-Math-7B-Instruct`，v2 主 classifier 应继续针对这个 frozen model 训练。hidden-state probe 通常不应期待跨 generator checkpoint 直接泛化。

可选扩展：

| 生成模型 | 建议 |
|---|---|
| Qwen2.5-Math-7B-Instruct | 主线，保持当前可比性 |
| Qwen2.5-Math-1.5B-Instruct | 可用于产生更多错误，但 CoT 质量较弱，不建议混入主训练，除非训练 multi-model probe |
| Qwen2.5-Math-72B-Instruct | 可作为高质量 holdout/teacher，不适合作为第一阶段数据主力 |
| Qwen3 或其它模型 | 需要重新提取 hidden states，建议单独建 probe，不要直接混在 v2 主线 |

### 5.2 Feature 升级

当前 feature 是四层 step-end hidden state concat。v2 建议保留这个强 baseline，同时增加以下候选：

| Feature | 动机 |
|---|---|
| all-layer 或更多层采样 | 当前 layer 20/27 最强，但 layer interaction 可能更好 |
| learned layer pooling | 避免手工固定 `[6,13,20,27]` |
| last-k-token pooling | step-end token 可能不足以表达整步状态 |
| step-start 与 step-end delta | 捕捉当前 step 对模型内部状态的变化 |
| residual delta `h_l - h_{l-1}` | 强化层间推理更新信号 |
| logprob / entropy / top-token margin | 置信度辅助特征，但不能单独依赖 |
| normalized step index / step length | 作为显式 covariate 加入，并做 ablation 防 shortcut |
| text TF-IDF 或轻量 text encoder | 与 hidden probe 做 late fusion，主要用于 hard negative 解释 |

重要约束：必须做 length-matched 和 step-index-matched evaluation。当前 step length baseline 已经很强，v2 不能只学到“长 step 更危险”。

## 6. Classifier 架构设计

### 6.1 第一阶段：强 MLP/fusion baseline

当前 MLP 是 `14336 -> 512 -> 128 -> 1`，数据量相对参数量偏小。建议先做一个更稳的 layer-wise encoder：

```text
per-layer hidden h_l
  -> per-layer LayerNorm + Linear(3584 -> 256)
  -> learned layer attention / mean-max pooling
  -> concat scalar features: logprob, length, step index, token count
  -> residual MLP: 512 -> 256 -> 64
  -> outputs:
       fhis_logit
       optional final_wrong_logit
       optional error_type_logits
```

优势：

- 参数量比直接 14336 输入 MLP 更可控。
- 可以学习哪些层对不同题型更重要。
- 兼容当前 `predict_scores` API。

### 6.2 第二阶段：trace-level causal hazard model

online 干预是一个 prefix-causal 问题：看到 Step 1..k 时判断 Step k 是否应触发。因此比单 step classifier 更自然的建模方式是 hazard model。

建议模型：

```text
每个 step 的 hidden/text/scalar vector
  -> step encoder
  -> causal Transformer 或 GRU over steps
  -> per-step hazard score p(FHIS at current step | previous accepted steps)
```

训练约束：

- wrong trace 中只有 FHIS step 为 positive，FHIS 前 steps 为 negative。
- correct trace 中所有 steps 为 negative。
- post-FHIS steps 可用于 auxiliary `after_invalid`，但不进入主 binary loss。
- 模型必须 causal mask，不能看未来 steps。

这个模型能学习“当前 step 相对前面轨迹是否异常”，有机会降低 pre-FHIS false stop。

### 6.3 排序模型和二阶段策略

当前 offline top-k coverage 不差，但 online threshold 差，说明排序信号比绝对概率更稳定。建议增加 pairwise/listwise ranking head：

```text
wrong trace: score(FHIS) > score(any pre-FHIS step) + margin
correct trace: all scores below calibrated null threshold
```

线上可采用二阶段：

| 阶段 | 目标 |
|---|---|
| high-recall router | 找出可疑 step，允许较低 precision |
| precision gate / verifier | 决定是否真的 retry 或 Lean-veto |

如果没有外部 verifier，probe-only retry 应使用更保守的 precision gate，而不是直接使用 high-recall router 阈值。

## 7. 训练方法

### 7.1 Split 和 sampling

必须保持：

- problem-disjoint split。
- dataset-disjoint holdout。
- 按 subject/difficulty/final correctness/FHIS position 分层抽样。
- 同一 problem 的所有 samples 只能出现在同一 split。

DataLoader 不建议按 row IID 随机采样。更好的方式：

- batch 按 trace 组织，包含一个 wrong trace 的 pre-FHIS negatives + FHIS positive。
- 每个 batch 混入 correct trace hard negatives。
- 对当前模型高分 FP 做 oversampling。

### 7.2 Loss 设计

当前 recall-biased loss 会把 threshold 推得很低，导致 high recall 但大量 false stop。v2 建议组合：

| Loss | 用途 |
|---|---|
| weighted BCE / focal loss | 处理 14% positive imbalance |
| pairwise ranking loss | 直接优化 FHIS 在 trace 内排第一 |
| listwise softmax loss | 优化 per-trace top-k coverage |
| correct-trace max-score penalty | 压低正确 trace 的最高风险分 |
| calibration loss / temperature scaling | 让分数可以作为 online 阈值使用 |
| auxiliary error-type loss | 促进模型学到错误结构，不只学长度 |

推荐主 loss：

```text
L = BCE_or_focal
  + lambda_rank * max(0, margin - score_fhis + score_prefhis)
  + lambda_correct * max(0, max_score_correct_trace - tau_null)
  + lambda_aux * auxiliary_losses
```

### 7.3 Hard negative mining loop

建议每轮训练后运行：

1. 在 validation/test-like pool 上找 top false positives。
2. 将 false positives 分成 long correct step、benign gap、notation issue、actually mislabeled 等类别。
3. 对 actually mislabeled 样本重标；对真实 hard negatives 加权加入下一轮训练。
4. 记录 false positive 的 subject、step index、length、error-looking pattern。

这个循环比单纯扩大随机数据更可能改善 online false stop。

### 7.4 阈值校准

不要再只使用 `positive_recall_target=0.99` 和 `negative_accuracy_floor=0.80` 选阈值。阈值应按具体 online policy 校准：

| 阈值类型 | 用途 |
|---|---|
| budget threshold | 固定 routed/verified step rate，例如 10%、20%、30% |
| recall threshold | 固定 FHIS recall，例如 0.80、0.90、0.95 |
| precision threshold | probe-only retry 时使用，限制 false stop |
| per-domain threshold | 如果 holdout 显示明显 domain shift，可按 dataset/subject 校准 |

最终 artifact 应保存多组 thresholds，而不是只保存一个 `decision_threshold`。

## 8. 需要的数据量

当前只有 1192 个 positive FHIS rows。对于 14336 维 hidden feature 和 MLP，这个规模只够证明 signal，不足以稳定训练可部署 classifier。

建议分阶段扩容：

| 阶段 | clean traces | wrong FHIS positives | step rows | 用途 |
|---|---:|---:|---:|---|
| v1.1 | 4000-6000 | 2500-3500 | 18000-30000 | 快速验证 hard negatives、ranking loss |
| v2 main | 18000-25000 | 10000-14000 | 70000-120000 | 训练主 classifier 和 sequence model |
| v2 robust | 40000+ | 20000+ | 150000+ | 多数据集、多题型、多温度稳定泛化 |

按当前 clean 数据比例估算：

- 当前每 2128 clean traces 产生 1192 positive FHIS，约 0.56 positive trace / clean trace。
- 若每题生成 4 条 samples，则每题约 2.2 条 clean wrong FHIS traces。
- 要达到 10000 positive FHIS，大约需要 18000 clean traces，约 4500-6000 道题，具体取决于新数据集错误率和过滤率。

Gold evaluation set 建议至少：

| Split | wrong FHIS traces | correct traces | 说明 |
|---|---:|---:|---|
| in-domain dev | >= 500 | >= 500 | 调阈值 |
| in-domain test | >= 1000 | >= 1000 | 冻结后报告 |
| MATH Level 5 holdout | >= 500 | >= 500 | 当前 22 太小 |
| second external holdout | >= 300 | >= 300 | Minerva/AIME/其它 |

## 9. 评估协议

v2 报告必须同时包含 offline 和 online-style 指标。

Offline row metrics：

- AUROC, AUPRC。
- F1 / balanced accuracy 只作为辅助，不作为主目标。
- ECE / reliability curve，检查分数校准。

Trace ranking metrics：

- recall@1, recall@2。
- top 10% / 20% / 30% budget coverage。
- FHIS rank percentile。

Online prefix metrics：

| 指标 | 含义 |
|---|---|
| FHIS step recall | FHIS step 被 flagged 的比例 |
| observable non-FHIS step FPR | 正确 trace steps + wrong trace pre-FHIS steps 的误触发率 |
| correct-trace false stop rate | 正确解被任何 step 误停的比例 |
| pre-FHIS stop rate on wrong traces | 真 FHIS 之前就误停的比例 |
| average stop step | 线上计算/生成成本 |
| solve rate after retry | 真正 test-time 干预收益 |
| abstention rate | router 是否过度保守 |

必须做 ablation：

- hidden only vs hidden + scalar。
- layer 20/27 only vs learned layer pooling。
- MLP vs sequence hazard model。
- BCE vs ranking loss。
- random negatives vs hard negatives。
- in-domain vs cross-dataset holdout。

## 10. 推荐实施路线

### Phase 0: 诊断 v1

产出：

- 当前 MLP/logistic 在 val/test 的 FP/FN 样本清单。
- FP 按 step length、step index、subject、error-looking text pattern 的统计。
- `first_invalid_step` 分布和 label disagreement 清单。

验收：

- 明确 v1 的主要误杀类型。
- 找出至少 200-500 条 hard negative candidates。

### Phase 1: 标注和数据 v1.1

产出：

- 扩展到 2500-3500 positive FHIS。
- 复标 rough/Codex 冲突、v1 FP/FN、Step 1 FHIS。
- 规范化 `error_type` taxonomy。

验收：

- MATH Level 5 holdout 至少 200 wrong traces。
- hard-negative eval 上 false positive 明显下降。

### Phase 2: 训练强 baseline

产出：

- layer-wise MLP/fusion classifier。
- ranking loss 版本。
- 多阈值 calibration artifact。

验收：

- in-domain AUPRC >= 0.60。
- top 30% coverage >= 0.93。
- FHIS recall 0.90 时 step FPR 相比 v1 至少下降 30%。

### Phase 3: 主数据扩容和 sequence model

产出：

- 10000+ positive FHIS。
- 70000+ step rows。
- causal hazard model。
- dataset-disjoint holdout 报告。

验收：

- in-domain AUPRC >= 0.65。
- MATH Level 5 holdout 至少 500 wrong traces，AUPRC 明显高于 step length/text baselines。
- probe-only retry 的 correct false stop rate 降到可接受区间。

### Phase 4: Online A/B

产出：

- no-router baseline。
- route-to-verifier policy。
- probe-only retry policy。
- verifier-guided repair policy。

验收：

- 在固定生成预算下，rough solve rate 高于 no-router。
- 在固定 verifier budget 下，FHIS coverage 高于 current MLP。
- 不因 false positive 造成大量 abstention。

## 11. 具体工程改动建议

建议新增或修改：

| 路径 | 内容 |
|---|---|
| `classifier/` | v2 计划、实验记录、误差分析表 |
| `data_generation/qwen25_fhis/schema/` | step-level v2 label schema |
| `src/fhis/train_probe.py` 或新模块 | ranking loss、layer-wise encoder、sequence hazard model |
| `src/fhis/metrics.py` | online prefix metrics、calibration metrics |
| `src/fhis/calibrate_router.py` | 输出多阈值表，而非单一 threshold |
| `data_generation/qwen25_fhis/scripts/` | hard-negative mining、label adjudication、dataset manifest |

模型 artifact 建议保存：

```json
{
  "model": "...",
  "probe_kind": "layerwise_mlp|hazard_transformer",
  "feature_spec": {
    "layers": [6, 13, 20, 27],
    "pooling": "step_end|last_k|layer_attention",
    "scalar_features": ["step_index", "step_length", "mean_logprob"]
  },
  "thresholds": {
    "budget_10pct": 0.0,
    "budget_20pct": 0.0,
    "fhis_recall_90": 0.0,
    "precision_gate": 0.0
  },
  "training_data": {
    "datasets": [],
    "problems": 0,
    "traces": 0,
    "positive_fhis_steps": 0,
    "negative_steps": 0
  }
}
```

## 12. 最重要的判断

当前 v1 的主要价值是证明 hidden-state FHIS signal 存在；它不是最终 online intervention classifier。下一版最应该优先解决的不是“换一个更大的 MLP”，而是：

1. 扩展并清洗数据，尤其是 hard negatives 和 cross-dataset holdout。
2. 把训练目标从 row classification 改成 trace ranking / prefix hazard。
3. 把阈值校准从单一 recall-biased threshold 改成 policy-specific threshold curves。
4. 用 online-style metrics 做早停和模型选择。

如果只能先做一个低成本改进，推荐顺序是：先 hard-negative mining + 复标，再加 ranking loss，最后再尝试更复杂的 sequence hazard model。
