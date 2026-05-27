#!/usr/bin/env python3
"""parse_cot.py

从 `cot_outputs.jsonl` 中解析每条 chain 的步骤（优先识别方括号格式 `[Step k]` 和
`[Final Answer]`），并把结构化的步块写入 `cot_steps.jsonl`（每行一条 chain）。

输出字段示例:
{
  "id": "problem id",
  "question": "...",
  "chain_index": 1,
  "label": "...",
  "provider": "deepseek",
  "model": "deepseek-v4-pro",
  "original_text": "...",
  "steps": [{"step_index":1, "text":"..."}, ...],
  "final_answer": "..."
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


# Prefer bracketed markers like [Step 1] and [Final Answer]
BRACKET_STEP_RE = re.compile(r"\[Step\s*(\d+)\]\s*(.*?)(?=(?:\n\[Step\s*\d+\])|\n\[Final\s*Answer\]|\Z)", re.I | re.S)
BRACKET_FINAL_RE = re.compile(r"\[Final\s*Answer\]\s*(.*)", re.I | re.S)

# Fallback: colon-style markers 'Step 1:' and 'Final Answer:'
COLON_STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*(\d+)\s*:\s*(.*?)(?=(?:\n\s*Step\s*\d+\s*:)|\n\s*Final\s*Answer\s*:|\Z)", re.I | re.S)
COLON_FINAL_RE = re.compile(r"Final\s*Answer\s*[:\-]?\s*(.*)", re.I | re.S)


def parse_chain_text(text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """解析单条 chain 文本，返回 (steps, final_answer).

    steps: list of {step_index: int, text: str}
    final_answer: str or None
    """
    if not text:
        return [], None

    # 1) 优先识别方括号格式
    m_iter = list(BRACKET_STEP_RE.finditer(text))
    steps: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None
    if m_iter:
        for m in m_iter:
            try:
                idx = int(m.group(1))
            except Exception:
                idx = len(steps) + 1
            content = m.group(2).strip()
            steps.append({"step_index": idx, "text": content})
        mf = BRACKET_FINAL_RE.search(text)
        if mf:
            final_answer = mf.group(1).strip()
        return steps, final_answer

    # 2) 回退到冒号格式
    m_iter = list(COLON_STEP_RE.finditer(text))
    if m_iter:
        for m in m_iter:
            try:
                idx = int(m.group(1))
            except Exception:
                idx = len(steps) + 1
            content = m.group(2).strip()
            steps.append({"step_index": idx, "text": content})
        mf = COLON_FINAL_RE.search(text)
        if mf:
            final_answer = mf.group(1).strip()
        return steps, final_answer

    # 3) 最后回退：按非空行拆分为若干步
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], None
    steps = [{"step_index": i + 1, "text": ln} for i, ln in enumerate(lines)]
    return steps, None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description='Parse CoT outputs into step blocks')
    parser.add_argument('--input', help='input cot_outputs.jsonl (one JSON per problem).')
    parser.add_argument('--run_id', help='run id to infer default input/out under lean_single_step/experiments/runs/<run_id>')
    parser.add_argument('--outdir', help='output directory (defaults to same dir as input)')
    parser.add_argument('--outfile', default='cot_steps.jsonl', help='output filename (default cot_steps.jsonl)')
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
    elif args.run_id:
        in_path = Path('lean_single_step') / 'experiments' / 'runs' / args.run_id / 'cot_outputs.jsonl'
    else:
        parser.error('must provide --input or --run_id')

    if not in_path.exists():
        raise FileNotFoundError(f'Input file not found: {in_path}')

    out_dir = Path(args.outdir) if args.outdir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.outfile

    items = read_jsonl(in_path)
    results: List[Dict[str, Any]] = []
    problems = 0
    chains = 0
    steps_total = 0

    for it in items:
        problems += 1
        pid = it.get('id')
        question = it.get('question')
        for ch in it.get('chains', []):
            chains += 1
            text = ch.get('text', '')
            parsed_steps, final = parse_chain_text(text)
            steps_total += len(parsed_steps)
            entry = {
                'id': pid,
                'question': question,
                'chain_index': ch.get('chain_index'),
                'label': ch.get('label'),
                'provider': ch.get('provider'),
                'model': ch.get('model'),
                'original_text': text,
                'steps': parsed_steps,
                'final_answer': final,
            }
            results.append(entry)

    write_jsonl(results, out_path)
    print(f'Parsed {problems} problems, extracted {chains} chains, {steps_total} steps -> {out_path}')


if __name__ == '__main__':
    main()
