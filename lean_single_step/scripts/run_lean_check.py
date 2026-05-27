#!/usr/bin/env python3
"""run_lean_check.py

简化版 Lean 验证工具：**仅使用 `lake env lean`** 检查生成的 `.lean` 文件。
自动跳过空白/占位符文件，提供清晰的成功/失败报告。

用法：
  # 通过 run_id 自动推导路径
  python scripts/run_lean_check.py --run_id test_run

  # 手动指定输入/输出
  python scripts/run_lean_check.py --input lean_generated --output verification.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 120) -> Dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        return {
            'returncode': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'timeout': False,
            'elapsed': elapsed,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            'returncode': None,
            'stdout': '',
            'stderr': '',
            'timeout': True,
            'elapsed': elapsed,
        }
    except FileNotFoundError:
        elapsed = time.time() - start
        return {
            'returncode': None,
            'stdout': '',
            'stderr': 'lean or lake command not found',
            'timeout': False,
            'elapsed': elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'returncode': None,
            'stdout': '',
            'stderr': str(e),
            'timeout': False,
            'elapsed': elapsed,
        }


def is_trivial_lean(file_path: str) -> bool:
    """检查文件内容是否为无实际证明代码的空壳/占位符。"""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
    except Exception:
        return True
    lines = content.splitlines()
    meaningful = [l.strip() for l in lines if l.strip() and not l.strip().startswith('--')]
    if not meaningful:
        return True
    if 'LLM returned empty' in content or 'LLM error' in content:
        return True
    return False


def copy_to_project_temp(src: str, project_dir: str) -> str:
    uid = uuid.uuid4().hex[:8]
    dst_dir = Path(project_dir) / '.lean_check_tmp'
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f'temp_check_{uid}.lean'
    shutil.copy(src, dst)
    return str(dst)


def remove_project_temp(path: str) -> None:
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Lean checks using only lake env lean')
    parser.add_argument('--run_id', help='实验 run_id，自动推导 input/output 路径')
    parser.add_argument('--input', default=None, help='目录或文件（.lean）')
    parser.add_argument('--output', default=None, help='输出 JSON 路径')
    parser.add_argument('--timeout', type=int, default=120, help='单文件检查超时（秒）')
    parser.add_argument('--project-dir', default=str(Path.home() / 'my_project'),
                        help='已编译 mathlib 的 Lean 项目路径（默认 ~/my_project）')
    args = parser.parse_args()

    if args.run_id:
        base = Path('lean_single_step') / 'experiments' / 'runs' / args.run_id
        if args.input is None:
            args.input = str(base / 'lean_generated')
        if args.output is None:
            args.output = str(base / 'verification.json')

    if not args.input or not args.output:
        parser.error('必须提供 --input 和 --output，或使用 --run_id')

    in_path = Path(args.input)
    files: List[str] = []
    if in_path.is_file():
        files = [str(in_path)]
    elif in_path.is_dir():
        files = sorted(glob.glob(os.path.join(str(in_path), '*.lean')))
    else:
        files = sorted(glob.glob(args.input))

    project_dir = args.project_dir
    results = []

    for f in files:
        result = {'file': f}
        if is_trivial_lean(f):
            result['ok'] = False
            result['error'] = 'Trivial or empty Lean code (placeholder/empty response)'
            results.append(result)
            continue

        temp_path = copy_to_project_temp(f, project_dir)
        try:
            cmd_res = run_cmd(['lake', 'env', 'lean', temp_path], cwd=project_dir, timeout=args.timeout)
        finally:
            remove_project_temp(temp_path)

        if cmd_res.get('timeout'):
            result['ok'] = False
            result['error'] = 'Timeout'
        elif cmd_res.get('returncode') == 0:
            result['ok'] = True
        else:
            result['ok'] = False
            result['error'] = cmd_res.get('stdout') or cmd_res.get('stderr') or 'Unknown error'
        results.append(result)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    ok_count = sum(1 for r in results if r.get('ok'))
    print(f'Wrote {len(results)} results to {args.output} ({ok_count} OK, {len(results)-ok_count} failed)')


if __name__ == '__main__':
    main()