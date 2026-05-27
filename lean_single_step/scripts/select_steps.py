#!/usr/bin/env python3
"""
select_steps.py

从解析好的 cot_steps.jsonl 中随机抽取部分步骤，生成 steps_selected.jsonl。

用法:
  # 通过 run_id 自动定位
  python select_steps.py --run_id test_run_05261304 --k 2

  # 或手动指定输入/输出
  python select_steps.py --input ... --out ... --k 2
"""
import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', help='实验 run_id，自动从 experiments/runs/<run_id>/ 读取输入并输出到同一目录')
    parser.add_argument('--input', help='输入 cot_steps.jsonl 路径（优先级高于 --run_id）')
    parser.add_argument('--out', help='输出 steps_selected.jsonl 路径（优先级高于 --run_id 自动生成）')
    parser.add_argument('--k', type=int, default=2, help='每条 chain 选取的步骤数')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 确定输入文件
    in_path = None
    if args.input:
        in_path = Path(args.input)
    elif args.run_id:
        in_path = Path('lean_single_step') / 'experiments' / 'runs' / args.run_id / 'cot_steps.jsonl'
    else:
        parser.error('必须提供 --input 或 --run_id')

    if not in_path.exists():
        raise FileNotFoundError(f'输入文件不存在: {in_path}')

    # 确定输出文件
    out_path = None
    if args.out:
        out_path = Path(args.out)
    elif args.run_id:
        out_path = Path('lean_single_step') / 'experiments' / 'runs' / args.run_id / 'steps_selected.jsonl'
    else:
        out_path = in_path.parent / 'steps_selected.jsonl'

    random.seed(args.seed)
    with in_path.open('r', encoding='utf-8') as f:
        chains = [json.loads(line) for line in f if line.strip()]

    selected = []
    for chain in chains:
        steps = chain.get('steps', [])
        if not steps:
            continue
        chosen = random.sample(steps, min(args.k, len(steps)))
        for step in chosen:
            selected.append({
                'id': chain['id'],
                'question': chain.get('question'),
                'chain_id': chain['chain_index'],
                'step_id': step['step_index'],
                'target_step': step['text'],
                'previous_steps': [s['text'] for s in steps if s['step_index'] < step['step_index']]
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Selected {len(selected)} steps -> {out_path}')

if __name__ == '__main__':
    main()