#!/usr/bin/env python3
"""
generate_cot.py

Chain-of-Thought generation using an LLM (OpenAI via llm_client) or a
deterministic mock generator.  Now uses concurrent batch calls to speed up
generation of multiple chains.
"""
from __future__ import annotations
import argparse, json, os, time, random
from pathlib import Path
from typing import List, Dict, Any
import re
from llm_client import call_llm, call_llm_batch


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    with p.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(items: List[Dict[str, Any]], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + '\n')


def load_prompt() -> str:
    here = Path(__file__).resolve().parent
    prompt_file = here.parent / 'prompts' / 'cot_prompt.md'
    if prompt_file.exists():
        return prompt_file.read_text(encoding='utf-8')
    return (
        "You are a strong mathematical reasoning model. For the given problem, produce a chain of thought:\n"
        "Format each chain as:\nChain N:\nStep 1: ...\nStep 2: ...\n...\nFinal Answer: <answer>\n"
    )


def generate_cot_mock(question: str, chain_idx: int, max_steps: int = 5) -> str:
    random.seed(hash(question) ^ chain_idx)
    step_count = random.randint(3, max_steps)
    pieces = []
    pieces.append(f"[Step 1] Restate the problem briefly: '{question[:120]}'\n")
    for s in range(2, step_count + 1):
        pieces.append(f"[Step {s}] Reasoning step {s} (heuristic placeholder).\n")
    pieces.append("[Final Answer] [placeholder]\n")
    return ''.join(pieces)


def generate_for_item(item: Dict[str, Any], n_chains: int,
                      prompt_base: str, use_mock: bool,
                      model: str) -> Dict[str, Any]:
    q = item.get('question') or item.get('text') or str(item.get('raw', ''))
    chains: List[Dict[str, Any]] = []

    if use_mock:
        for i in range(n_chains):
            text = generate_cot_mock(q, i)
            chains.append({
                'chain_index': i + 1,
                'label': f"{item.get('id') or 'unknown'}__chain{i+1}",
                'provider': 'mock',
                'model': 'mock',
                'text': text,
            })
        return {
            "id": item.get('id'),
            "question": q,
            "chains": chains,
            "model": 'mock',
        }

    # Build the system prompt
    system_prompt = (
        prompt_base
        + "\n\nNote: Produce exactly ONE chain for this request. Do not add chain numbers or problem ids."
        + "\nIMPORTANT: Format each step using square-bracket markers like [Step 1], [Step 2], ... and finish with [Final Answer]."
    )
    user_prompt = f"Problem: {q}\n\nPlease produce ONE chain as described."

    # Prepare tasks for concurrent batch
    tasks = []
    for i in range(n_chains):
        tasks.append({
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'model': model,
            'mock': False,
            'retries': 2,
            'timeout': 120,
            'reasoning': False,   # you can change to True if needed
            'call_label': f"chain-{i+1}",
        })

    # Concurrently call the LLM
    raw_texts = call_llm_batch(tasks, max_workers=min(n_chains, 5))

    # Process each response
    for i, text in enumerate(raw_texts):
        provider = 'openai'
        used_model = model
        # Formatting validation: if missing markers, try reformatting
        if not re.search(r"\[Step\s*\d+\]|\[Final\s*Answer\]", text, re.IGNORECASE):
            try:
                fmt_system = (
                    "You are a formatter. Reformat the following chain so that each step "
                    "is marked with [Step N], ..., ending with [Final Answer]. "
                    "Preserve the original content. Only reformat."
                )
                fmt_user = f"Original output:\n\n{text}\n\nPlease reformat exactly as requested."
                formatted = call_llm(
                    system_prompt=fmt_system,
                    user_prompt=fmt_user,
                    model=model,
                    mock=False,
                    retries=1,
                    timeout=60,
                    reasoning=False,
                    call_label=f"fmt-{i+1}",
                )
                if re.search(r"\[Step\s*\d+\]|\[Final\s*Answer\]", formatted, re.IGNORECASE):
                    text = formatted
            except Exception:
                pass

        chains.append({
            'chain_index': i + 1,
            'label': f"{item.get('id') or 'unknown'}__chain{i+1}",
            'provider': provider,
            'model': used_model,
            'text': text,
        })

    return {
        "id": item.get('id'),
        "question": q,
        "chains": chains,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate CoT chains for sampled problems')
    parser.add_argument('--input', required=True, help='input sampled JSONL (from sample_data.py)')
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--chains', type=int, default=5)
    parser.add_argument('--outdir', default=None, help='optional output dir (default: experiments/runs/<run_id>)')
    parser.add_argument('--mock', action='store_true', help='use deterministic mock CoT generator')
    parser.add_argument('--model', default=None, help='model id to use (default: OPENAI_MODEL env or gpt-4o)')
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_base = Path(args.outdir) if args.outdir else Path('lean_single_step') / 'experiments' / 'runs' / args.run_id
    out_base.mkdir(parents=True, exist_ok=True)

    selected_model = args.model or os.environ.get('OPENAI_MODEL') or 'gpt-4o'
    prompt_base = load_prompt()
    items = read_jsonl(in_path)

    run_cfg = {
        'run_id': args.run_id,
        'model': selected_model,
        'chains': args.chains,
        'mock': bool(args.mock),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    with open(out_base / 'run_config.json', 'w', encoding='utf-8') as fh:
        json.dump(run_cfg, fh, indent=2, ensure_ascii=False)

    outputs = []
    for idx, it in enumerate(items):
        out = generate_for_item(it, args.chains, prompt_base, use_mock=args.mock, model=selected_model)
        outputs.append(out)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(items)}")

    out_file = out_base / 'cot_outputs.jsonl'
    write_jsonl(outputs, out_file)
    write_jsonl(items, out_base / 'inputs.jsonl')
    print(f"Wrote {len(outputs)} CoT outputs to {out_file}")


if __name__ == '__main__':
    main()