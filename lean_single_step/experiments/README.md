# experiments/

每次实验的运行输出保存在 `experiments/runs/<run_id>/`。
目录结构：
- inputs.jsonl: 原始题目输入
- cot_outputs.jsonl: CoT 输出（每题多条链）
- steps_selected.jsonl: 被选为形式化目标的步骤
- generated_lean/: 生成的 .lean 文件
- lean_logs/: 单文件编译日志
- metrics.json: 本次运行计算的指标
