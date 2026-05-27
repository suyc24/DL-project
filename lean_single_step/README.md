# 项目说明：Chain-of-Thought 到 Lean 形式化验证流水线

本项目实现了一套从自然语言数学问题出发，生成推理链（Chain-of-Thought），将推理步骤转化为 Lean 4 定理，并自动编译验证的完整流水线。整体流程由多个 Python 脚本串联，支持批量并发调用大语言模型（LLM），并通过与 Lean 编译器的交互实现代码修正。

---

## 整体工作流（Workflow）

```
┌──────────────┐
│  数学数据集    │
│ (MATH/miniF2F)│
└──────┬───────┘
       │ sample_data.py
       ▼
┌─────────────────────┐
│  采样后的 JSONL      │  (每行: {"id":..., "question":...})
└──────┬──────────────┘
       │ generate_cot.py
       ▼
┌───────────────────────────────────────────────┐
│  cot_outputs.jsonl                            │
│  (每行一个问题，包含多条 chain，每条含推理文本) │
└──────┬────────────────────────────────────────┘
       │ parse_cot.py
       ▼
┌──────────────────────────────────┐
│  cot_steps.jsonl                │
│  (每条 chain 拆分出各个推理步骤)  │
└──────┬───────────────────────────┘
       │ select_steps.py
       ▼
┌───────────────────────────────────────────────┐
│  steps_selected.jsonl                         │
│  (随机选择的目标步骤及其前序步骤，用于生成定理) │
└──────┬────────────────────────────────────────┘
       │ generate_lean.py
       ▼
┌───────────────────────────────────────────────┐
│  lean_generated/ 目录                         │
│  生成 .lean 文件，并附带编译修复后的最终代码     │
└──────┬────────────────────────────────────────┘
       │ run_lean_check.py
       ▼
┌────────────────────────────┐
│  verification.json         │
│  (每个 .lean 文件的编译结果) │
└────────────────────────────┘
```

流水线通过 `--run_id` 统一管理实验，所有中间产物和最终结果均存放在 `lean_single_step/experiments/runs/<run_id>/` 目录下。

---

## 目录结构（典型实验输出）

```
lean_single_step/
  experiments/
    runs/
      <run_id>/
        inputs.jsonl              # 采样输入（由 generate_cot 自动保存）
        run_config.json           # 运行配置
        cot_outputs.jsonl         # 含推理链的原始 LLM 输出
        cot_steps.jsonl           # 解析后的推理步骤
        steps_selected.jsonl      # 选中的步骤（每条链选 k 个）
        lean_generated/           # 生成的 .lean 文件（及响应文本）
        verification.json         # 编译验证报告
```

---

## 核心组件详解

### 1. `sample_data.py` – 数据采样与标准化

**功能**  
从 MATH 数据集、miniF2F 数据集或通用 JSONL/JSON/TXT 文件中采样指定数量的数学问题，输出统一格式的 JSONL。

**输入支持**  
- 本地 MATH 目录（递归读取 `.json` 文件）  
- Hugging Face 数据集名（如 `hendrydong/hendrycks_math`，需安装 `datasets`）  
- miniF2F JSONL 文件（字段包含 `question`/`src`）  
- 通用文本文件（每行一题）  

**输出**  
每条数据为 `{"id": "...", "question": "..."}` 的 JSONL。

**用法示例**  
```bash
# 从 Hugging Face 加载 MATH 训练集，采样 100 条
python sample_data.py --input hendrydong/hendrycks_math --n 100 --out data/math_100.jsonl --split train

# 从本地目录加载 MATH
python sample_data.py --input /path/to/MATH --n 50 --out data/math_50.jsonl

# 加载 miniF2F
python sample_data.py --input miniF2F.jsonl --dataset minif2f --n 30 --out data/minif2f_30.jsonl

# 通用文本文件
python sample_data.py --input problems.txt --n 20 --out data/sampled.jsonl
```

---

### 2. `generate_cot.py` – 批量生成推理链

**功能**  
对每个问题调用 LLM 生成多条 Chain-of-Thought 推理链（默认 5 条），支持 Mock 模式快速测试。

**关键参数**  
- `--input`：输入采样 JSONL  
- `--run_id`：实验 ID，决定输出目录  
- `--chains`：每个问题生成链的数量（默认 5）  
- `--model`：LLM 模型（默认 `gpt-4o` 或环境变量 `OPENAI_MODEL`）  
- `--mock`：使用确定性模拟生成，无需 API 调用  

**并发策略**  
内部通过 `call_llm_batch` 并发请求 LLM，最大并发数可配置（代码中 `min(n_chains, 5)`，可修改）。

**输出**  
`cot_outputs.jsonl`，每行包含一个问题的所有 chain 及其元信息。

**用法示例**  
```bash
# 真实调用（需配置 OPENAI_API_KEY）
python generate_cot.py --input data/sampled.jsonl --run_id exp1 --model gpt-4o

# Mock 测试
python generate_cot.py --input data/sampled.jsonl --run_id test_mock --mock
```

---

### 3. `parse_cot.py` – 解析推理步骤

**功能**  
从 `cot_outputs.jsonl` 中提取每条 chain 的步骤文本。  
支持方括号格式（`[Step 1] ... [Final Answer] ...`）和冒号格式（`Step 1: ... Final Answer: ...`），并提供回退拆分。

**输出**  
`cot_steps.jsonl`，每行为一条 chain 的解析结果，包含 `steps` 列表和 `final_answer`。

**用法示例**  
```bash
# 通过 run_id 自动定位输入
python parse_cot.py --run_id exp1

# 手动指定输入/输出
python parse_cot.py --input /path/to/cot_outputs.jsonl --outdir /path/to/out --outfile my_steps.jsonl
```

---

### 4. `select_steps.py` – 随机选取目标步骤

**功能**  
从每条 chain 中随机选择 `k` 个步骤（默认 2），并收集该步骤之前的所有步骤作为上下文。  
这些数据将用于让 LLM 将单个推理步骤形式化为 Lean 定理。

**输出**  
`steps_selected.jsonl`，每条包含 `target_step`、`previous_steps`（列表）、`chain_id`、`step_id` 等。

**用法示例**  
```bash
# 依赖 run_id
python select_steps.py --run_id exp1 --k 3 --seed 42

# 手动指定
python select_steps.py --input exp1/cot_steps.jsonl --out exp1/steps_selected.jsonl --k 2
```

---

### 5. `generate_lean.py` – 生成 Lean 代码并编译修复

**功能**  
对每个选中的步骤，构造提示词要求 LLM 生成一个 Lean 定理（过渡契约：假定前序步骤成立，证明当前步骤）。  
随后调用 `lake env lean` 进行编译，若失败则用错误信息反馈给 LLM 进行最多 `--max-rounds` 轮修复（默认 5 轮）。

**关键参数**  
- `--run_id`：实验 ID  
- `--model`：LLM 模型  
- `--project-dir`：已编译 Mathlib 的 Lean 项目路径（默认 `~/my_project`）  
- `--max-rounds`：最大修复轮数  

**输出**  
在 `lean_generated/` 目录下生成 `.lean` 文件，并附加 `.resp.txt` 记录 LLM 原始响应。  
文件命名规则：`{problem_id}__{chain_id}__step{step_id}.lean`。

**用法示例**  
```bash
python generate_lean.py --run_id exp1 --model deepseek-v3 --project-dir ~/mathlib4
```

---

### 6. `run_lean_check.py` – 编译验证报告

**功能**  
对 `lean_generated/` 中的所有 `.lean` 文件执行 `lake env lean` 编译，生成 JSON 验证报告，自动跳过空白或占位符文件。

**输出**  
`verification.json`，每个文件记录 `ok`（布尔值）和 `error`（失败信息）。

**用法示例**  
```bash
# 通过 run_id 自动定位
python run_lean_check.py --run_id exp1 --project-dir ~/mathlib4

# 手动指定输入目录和输出文件
python run_lean_check.py --input exp1/lean_generated --output exp1/verification.json --project-dir ~/mathlib4
```

---

### 7. `llm_client.py` – LLM 调用基础设施

本项目使用的 LLM 客户端封装，提供：

- `call_llm()`：同步单次调用，支持重试、超时、推理模式控制  
- `call_llm_batch()`：线程池并发调用，用于加速多链生成  
- 自动加载 `.env` 文件中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 等环境变量  
- 日志记录到 `llm_calls.log`（可配置）

需要用户提供有效的 OpenAI API Key（或兼容接口）。

---

## 环境准备与依赖

1. **Python 环境**  
   Python 3.10+，安装依赖：
   ```bash
   pip install openai datasets
   ```

2. **Lean 4 环境**  
   需要已安装 Lean 4 及 Mathlib4 的项目目录（例如通过 `lake` 创建）。  
   确保在项目目录中执行 `lake build` 成功编译 Mathlib。  
   所有编译检查均通过 `lake env lean` 进行，因此项目本身必须可用。

3. **API Key**  
   在项目根目录或脚本所在目录的 `.env` 文件中设置：
   ```
   OPENAI_API_KEY=your_key
   OPENAI_BASE_URL=https://api.openai.com/v1   # 可选，兼容其他服务
   OPENAI_MODEL=gpt-4o                          # 可选默认模型
   ```

---

## 完整运行示例

假设已准备好 Lean 项目路径 `~/my_project`，并且 API Key 已配置。

```bash
# 1. 采样数据
python sample_data.py \
  --input hendrydong/hendrycks_math \
  --n 20 \
  --out data/math20.jsonl \
  --split train

# 2. 生成推理链
python generate_cot.py \
  --input data/math20.jsonl \
  --run_id demo \
  --model gpt-4o

# 3. 解析步骤
python parse_cot.py --run_id demo

# 4. 选择步骤 (每条链选2个)
python select_steps.py --run_id demo --k 2

# 5. 生成 Lean 代码 (在项目环境中运行)
python generate_lean.py \
  --run_id demo \
  --model gpt-4o \
  --project-dir ~/my_project

# 6. 编译验证
python run_lean_check.py \
  --run_id demo \
  --project-dir ~/my_project

# 查看报告
cat lean_single_step/experiments/runs/demo/verification.json
```

---

## 自定义与扩展

- **修改 CoT 提示词**：编辑 `prompts/cot_prompt.md` 文件，`generate_cot.py` 会优先读取。
- **修改 Lean 生成模板**：编辑 `prompts/lean_single_step_template.md`，该文件需包含系统提示和用户模板（用 `---` 分隔）。
- **调整并发数量**：在 `generate_cot.py` 中 `call_llm_batch(tasks, max_workers=...)` 的 `max_workers` 目前硬编码为 `min(n_chains, 5)`，可根据服务器能力修改。
- **更换 LLM 后端**：`llm_client.py` 基于 OpenAI 兼容接口，若使用其他服务（如 DeepSeek），只需修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

---

## 注意事项

- Lean 编译依赖正确的项目环境，`generate_lean.py` 和 `run_lean_check.py` 均需传递 `--project-dir` 或使用默认 `~/my_project`。请确保该目录下 `lake env lean` 可以正常工作。
- 生成 Lean 代码时，LLM 可能会输出不符合格式的内容，`post_process_code` 会尝试添加 `import Mathlib`，但无法保证语义正确性。
- 流水线各步骤可通过 `--run_id` 串联，但也可以手动指定路径独立运行。
- 所有生成文件均为 UTF-8 编码，日志使用英文，控制台输出部分中文。
- 若使用 Hugging Face 数据集，需注意网络访问，必要时设置 `HF_ENDPOINT` 环境变量（如 `https://hf-mirror.com`）。