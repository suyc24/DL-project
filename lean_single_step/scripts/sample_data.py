#!/usr/bin/env python3
"""
sample_data.py

Flexible sampling & preprocessing for the chain‑to‑Lean pipeline.

Supports:
  - Generic JSONL / JSON / TXT files
  - MATH dataset: either local directory (as released on GitHub) or direct
    download from Hugging Face via `datasets` (e.g., `--input math`)
  - miniF2F dataset: JSONL file with `question`/`text`/`src` field

Outputs normalized JSONL with { "id": "...", "question": "..." }.
"""
from __future__ import annotations
import argparse, json, os, random, glob, hashlib
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# 通用加载器
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    return [data] if isinstance(data, dict) else []


def load_txt(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            t = line.strip()
            if not t:
                continue
            items.append({"question": t, "raw": t})
    return items


# ---------------------------------------------------------------------------
# MATH 数据集支持（本地目录）
# ---------------------------------------------------------------------------
def load_math_local(root_dir: Path) -> List[Dict[str, Any]]:
    """遍历 MATH 数据集的本地目录，收集所有 .json 文件。"""
    items = []
    for json_file in root_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding='utf-8'))
            problem_text = data.get("problem") or data.get("question") or ""
            if not problem_text:
                continue
            rel_path = json_file.relative_to(root_dir)
            uid = str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_')
            items.append({
                "id": uid,
                "question": problem_text,
                "source": "MATH",
                "raw": data,
            })
        except Exception:
            continue
    return items


def load_math_hf(dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    使用 Hugging Face `datasets` 库加载 MATH 数据集。
    数据集名称例如 `math`，split 可以是 `train` 或 `test`。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "`datasets` package is required to load MATH from Hugging Face. "
            "Install it with `pip install datasets`."
        )

    # 用户可通过环境变量 HF_ENDPOINT 设置镜像，例如 https://hf-mirror.com
    # export HF_ENDPOINT=https://hf-mirror.com
    ds = load_dataset(dataset_name, split=split)
    items = []
    for i, example in enumerate(ds):
        problem_text = example.get("problem") or example.get("question") or ""
        if not problem_text:
            continue
        # 使用 dataset 提供的 id（如有）或自动生成
        uid = example.get("id") or f"math_{split}_{i}"
        items.append({
            "id": str(uid),
            "question": problem_text,
            "source": "MATH",
            "raw": example,
        })
    return items


def load_math_dataset(input_str: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    根据输入字符串自动选择加载方式：
    - 若 input_str 是存在的目录，则调用本地加载器。
    - 否则，将其视为 Hugging Face 数据集名称，调用 `datasets` 加载。
    """
    path = Path(input_str).expanduser()
    if path.exists() and path.is_dir():
        return load_math_local(path)
    else:
        # 如果看起来像 HF 名称（包含 '/'），直接使用 HF 加载
        if '/' in input_str:
            return load_math_hf(input_str, split=split)
        else:
            raise FileNotFoundError(
                f"MATH input '{input_str}' is neither an existing directory "
                f"nor a valid Hugging Face dataset name (should contain '/')."
            )


# ---------------------------------------------------------------------------
# miniF2F 数据集支持
# ---------------------------------------------------------------------------
def load_minif2f(path: Path) -> List[Dict[str, Any]]:
    """加载 miniF2F 的 JSONL 文件，期望每行有 'id' 和 'question'/'text'/'src'。"""
    items = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = obj.get("question") or obj.get("text") or obj.get("src", "")
            if not q:
                continue
            uid = obj.get("id") or hashlib.sha1(q.encode()).hexdigest()[:8]
            items.append({
                "id": str(uid),
                "question": q,
                "source": "miniF2F",
                "raw": obj,
            })
    return items


# ---------------------------------------------------------------------------
# 采样与写入
# ---------------------------------------------------------------------------
def sample_items(items: List[Dict[str, Any]], n: int, seed: int | None) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
    if not items:
        return []
    if n >= len(items):
        random.shuffle(items)
        return items
    return random.sample(items, n)


def write_jsonl(items: List[Dict[str, Any]], out_path: Path) -> None:
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sample & normalize problems from various sources into a JSONL for the pipeline"
    )
    parser.add_argument('--input', required=True, help='file, directory, or HF dataset name (e.g., math)')
    parser.add_argument('--dataset', choices=['auto', 'math', 'minif2f', 'generic'], default='auto',
                        help='force dataset type (default: auto-detect)')
    parser.add_argument('--n', type=int, default=50, help='number of samples to draw')
    parser.add_argument('--out', required=True, help='output JSONL path')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--split', default='train', help='split for HF datasets (default: train)')
    args = parser.parse_args()

    # ---- 自动检测 dataset 类型 ----
    in_path = Path(args.input).expanduser()
    if args.dataset == 'auto':
        if not in_path.exists():
            # 可能是 HF 数据集名称
            if '/' in args.input and not any(in_path.suffix in ('.jsonl', '.json', '.txt') for _ in [in_path]):
                args.dataset = 'math'  # 假定包含 '/' 且不是本地文件即为 math 数据集
            else:
                parser.error(f"Cannot determine dataset type for '{args.input}'. Use --dataset to specify.")
        elif in_path.is_dir():
            # 目录：可能是 MATH 或通用
            sub_dirs = [p.name for p in in_path.iterdir() if p.is_dir()]
            if any(d in ('train', 'test') for d in sub_dirs):
                args.dataset = 'math'
            elif list(in_path.glob('*.json')):
                args.dataset = 'math'
            else:
                args.dataset = 'generic'
        elif in_path.suffix == '.jsonl':
            # JSONL：可能是 miniF2F 或通用
            try:
                with in_path.open('r', encoding='utf-8') as fh:
                    first = json.loads(fh.readline())
                if 'src' in first or 'tgt' in first:
                    args.dataset = 'minif2f'
                else:
                    args.dataset = 'generic'
            except Exception:
                args.dataset = 'generic'
        else:
            args.dataset = 'generic'

    # ---- 根据确定的类型加载数据 ----
    if args.dataset == 'math':
        items = load_math_dataset('hendrydong/hendrycks_math', args.split)
    elif args.dataset == 'minif2f':
        items = load_minif2f(in_path)
    else:
        # generic loader
        raw = []
        if in_path.is_dir():
            for ext in ("*.jsonl", "*.json", "*.txt"):
                for f in sorted(in_path.glob(ext)):
                    if f.suffix == ".jsonl":
                        raw.extend(load_jsonl(f))
                    elif f.suffix == ".json":
                        raw.extend(load_json(f))
                    else:
                        raw.extend(load_txt(f))
                if raw:
                    break
        else:
            if in_path.suffix == ".jsonl":
                raw = load_jsonl(in_path)
            elif in_path.suffix == ".json":
                raw = load_json(in_path)
            else:
                raw = load_txt(in_path)

        # Normalize
        items = []
        for i, obj in enumerate(raw):
            q = obj.get("question") or obj.get("problem") or obj.get("statement") or obj.get("text", "")
            if not q and isinstance(obj, str):
                q = obj
            uid = obj.get("id") or f"sample_{i}_{hashlib.sha1(q.encode()).hexdigest()[:8]}"
            items.append({
                "id": str(uid),
                "question": q,
                "source": args.input if in_path.is_dir() else in_path.name,
                "raw": obj,
            })

    if not items:
        print(f"No valid items found for input '{args.input}'")
        return

    sampled = sample_items(items, args.n, args.seed)
    write_jsonl(sampled, Path(args.out))
    print(f"Loaded {len(items)} items, sampled {len(sampled)} → {args.out}")


if __name__ == '__main__':
    main()