# Lean 单步形式化循环

这个目录是一个自包含的单步 Lean 形式化实验。

目标：

1. 让 AI 生成中文结构化 CoT；
2. 解析 CoT 中的 `[Step N]`；
3. 随机抽取若干步骤；
4. 让 AI 把抽到的单步形式化为 Lean 局部 transition contract；
5. 用 Lean 编译检查生成文件。

注意：这里评估的是“局部单步 transition”，不是整道题的完整证明。
Lean 在这里主要作为“单步前提义务提取器”：如果一步无法完全证明，允许把缺失数学事实显式暴露出来，帮助 AI 发现这一步是否跳得太大或发生幻觉。

## 目录

```text
lean_single_step_formalization/
  scripts/run_loop.py              # 端到端循环
  scripts/llm_client.py            # 自包含 LLM 调用；支持 openai 和 codex
  configs/default.json             # 本机路径、LLM、Lean 检查配置
  prompts/cot_prompt.md            # 中文 CoT prompt
  prompts/lean_step_contract.md    # 中文 Lean 单步 contract prompt
  data/smoke_problems.jsonl        # 本地 smoke 输入
  experiments/runs/<run_id>/       # 生成输出
```

## 本地 Mock Smoke

不调用 AI，也不跑 Lean：

```bash
python lean_single_step_formalization/scripts/run_loop.py \
  --config lean_single_step_formalization/configs/default.json \
  --input lean_single_step_formalization/data/smoke_problems.jsonl \
  --run-id smoke_mock \
  --chains 1 \
  --sample-steps 2 \
  --mock \
  --skip-lean-check
```

输出会在：

```text
lean_single_step_formalization/experiments/runs/<run_id>/
  input/problems.jsonl
  cot/cot_outputs.jsonl
  cot/cot_steps.jsonl
  selection/steps_selected.jsonl
  lean/*.lean
  lean/*.response.txt
  lean/lean_generation_manifest.jsonl
  verification/verification.json
  run_config.json
  run_summary.json
```

## 下载一小份 OlympiadBench

```bash
python lean_single_step_formalization/scripts/download_olympiadbench_sample.py \
  --limit 20 \
  --out lean_single_step_formalization/data/olympiadbench_non_geometry_20.jsonl
```

## LLM 后端 1：本地 Codex CLI

默认后端是本地 `codex exec`，使用本机 Codex 登录状态和配置，不在脚本里直接使用 API key。

先确认：

```bash
codex --help
```

运行：

```bash
python lean_single_step_formalization/scripts/run_loop.py \
  --config lean_single_step_formalization/configs/default.json \
  --input lean_single_step_formalization/data/olympiadbench_non_geometry_20.jsonl \
  --run-id codex_real \
  --chains 1 \
  --sample-steps 1 \
  --llm-timeout 900 \
  --max-workers 1 \
  --project-dir /root/mathlib4
```

建议 Codex CLI 模式先用 `--chains 1 --max-workers 1`，避免同时启动太多本地 Codex 进程。

## LLM 后端 2：OpenAI 兼容 API

设置环境变量：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # 可选
export OPENAI_MODEL=...      # 可选
```

运行：

```bash
python lean_single_step_formalization/scripts/run_loop.py \
  --input lean_single_step_formalization/data/smoke_problems.jsonl \
  --run-id openai_real \
  --llm-provider openai \
  --chains 3 \
  --sample-steps 2 \
  --project-dir /root/mathlib4
```

默认 Lean 项目路径优先读取 `LEAN_PROJECT_DIR`；如果本机存在 `/root/mathlib4`，会自动使用它；否则回退到项目内的 `lean_fhis`。

换机器时，优先改 [configs/default.json](/root/DL-project/lean_single_step_formalization/configs/default.json) 里的：

```json
{
  "paths": {"lean_project_dir": "/path/to/mathlib4"},
  "llm": {"provider": "codex", "codex_cwd": "/path/to/repo"}
}
```

## 需要人工检查的三层

- 编译层：`verification.json` 是否通过 Lean。
- 前提层：优先完全证明；其次接受局部 `h_missing_*` 假设；最后才接受 `obligation_*` axiom fallback。
- statement 层：theorem 不能是 `True`、不能空洞、不能把目标结论直接塞进假设或 axiom。
- step alignment 层：假设应来自前序步骤，或者明确写成缺失数学义务。

单纯 Lean 编译通过不等于步骤语义正确。

`verification.json` 会记录：

- `dependency_mode = complete`：没有显式缺失前提或全局 axiom。
- `dependency_mode = local_missing_hypotheses`：使用了 theorem 参数里的 `h_missing_*`。
- `dependency_mode = global_axiom_fallback`：使用了全局 `axiom obligation_*`。
- `kernel_axioms`：通过 `#print axioms <theorem>` 抽取 Lean kernel 层看到的 axiom 依赖。
