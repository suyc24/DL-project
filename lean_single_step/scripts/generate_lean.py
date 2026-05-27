#!/usr/bin/env python3
"""
generate_lean.py

Send selected target steps (with context) to the LLM, compile the result
with `lake env lean`, and use compiler errors to improve the code (up to 3 rounds).

Usage:
  python scripts/generate_lean.py --run_id test_run --model deepseek-v4-pro
"""
from __future__ import annotations
import argparse, json, os, re, logging, subprocess, uuid
from pathlib import Path
from llm_client import call_llm

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / 'prompts' / 'lean_single_step_template.md'
logger = logging.getLogger("llm_client")


# ---------------------------------------------------------------------------
# Lean compilation helper
# ---------------------------------------------------------------------------
def compile_lean(code: str, project_dir: str, timeout: int = 120) -> tuple[bool, str]:
    """Run `lake env lean` on a temporary file. Returns (success, output)."""
    tmp_dir = Path(project_dir) / '.lean_compile_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f'temp_{uuid.uuid4().hex[:8]}.lean'
    tmp_path.write_text(code, encoding='utf-8')
    try:
        proc = subprocess.run(
            ['lake', 'env', 'lean', str(tmp_path)],
            cwd=project_dir,
            capture_output=True, text=True,
            timeout=timeout
        )
        success = proc.returncode == 0
        output = (proc.stdout + proc.stderr).strip()
        return success, output
    except subprocess.TimeoutExpired as e:
        output = f"Timeout after {timeout} seconds"
        logger.warning(output)
        return False, output
    except Exception as e:
        output = f"Lean compilation error: {e}"
        logger.error(output)
        return False, output
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------------
def load_template() -> (str, str):
    """Return (system_prompt, user_template)."""
    if TEMPLATE_PATH.exists():
        content = TEMPLATE_PATH.read_text(encoding='utf-8')
        parts = content.split('\n---\n', 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    # fallback
    sys = (
        "You are a Lean 4 expert. Formalise ONLY the target step as a **transition contract**: "
        "a theorem that assumes the previous steps' conclusions and, if necessary, explicit "
        "`h_missing_*` hypotheses for missing lemmas. Output must be a single ```lean ``` fenced block."
    )
    usr = (
        "Previous steps:\n{PREVIOUS_STEPS}\n\nTarget step:\n{TARGET_STEP}\n\n"
        "Produce the theorem (include `import Mathlib` if needed)."
    )
    return sys, usr


def extract_lean_block(text: str) -> str:
    m = re.search(r"```\s*(?:lean4?|lean)\s*\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    # 容错处理
    m = re.match(r"```\s*(?:lean4?|lean)\s*\n(.*)", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def build_initial_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def add_feedback(messages: list[dict], assistant_code: str, error_output: str):
    messages.append({"role": "assistant", "content": assistant_code})
    messages.append({
        "role": "user",
        "content": (
            "Lean rejected the code with the following errors:\n\n"
            f"{error_output}\n\n"
            "Repair the code. Keep the theorem statement exactly, only modify the proof or definitions. "
            "Output the corrected code in a ```lean ``` fenced block."
        )
    })


def post_process_code(code: str) -> str:
    """Normalise imports and ensure `import Mathlib` exists."""
    code = extract_lean_block(code)
    lines = code.splitlines()
    if not any(line.strip().startswith('import') for line in lines):
        lines.insert(0, 'import Mathlib')
    has_mathlib = False
    cleaned = []
    for line in lines:
        s = line.strip()
        if s == 'import Mathlib':
            if not has_mathlib:
                cleaned.append(line)
                has_mathlib = True
        elif s.startswith('import Mathlib.'):
            if not has_mathlib:
                cleaned.append('import Mathlib')
                has_mathlib = True
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--input')
    parser.add_argument('--out_dir')
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--model', default=None)
    parser.add_argument('--project-dir', default=str(Path.home() / 'my_project'),
                        help='Lean project with compiled Mathlib')
    parser.add_argument('--max-rounds', type=int, default=5,
                        help='Max compilation‑repair rounds')
    args = parser.parse_args()

    run_id = args.run_id
    in_path = Path(args.input) if args.input else (
        Path('lean_single_step') / 'experiments' / 'runs' / run_id / 'steps_selected.jsonl'
    )
    out_base = Path(args.out_dir) if args.out_dir else (
        Path('lean_single_step') / 'experiments' / 'runs' / run_id / 'lean_generated'
    )
    if not in_path.exists():
        raise FileNotFoundError(f'输入文件不存在: {in_path}')
    out_base.mkdir(parents=True, exist_ok=True)

    system_prompt, user_template = load_template()
    selected_model = args.model or os.environ.get('OPENAI_MODEL') or 'gpt-4o'

    for item in read_jsonl(in_path):
        pid = item.get('id') or 'unknown'
        chain_id = item.get('chain_id') or 'chain'
        step_id = item.get('step_id') or 'step'
        previous = item.get('previous_steps') or []
        target = item.get('target_step') or item.get('step_text') or ''

        # Build initial user prompt
        prev = '\n'.join(previous) if previous else ''
        filled = user_template.replace('{PREVIOUS_STEPS}', prev)
        filled = filled.replace('{TARGET_STEP}', target)
        filled = filled.replace('{PROBLEM_ID}', str(pid))
        filled = filled.replace('{CHAIN_ID}', str(chain_id))
        filled = filled.replace('{STEP_ID}', str(step_id))
        filled = filled.replace('{FULL_CONTEXT}', prev + ('\n' + target if prev else target))

        if args.mock:
            lean_code = f"theorem mock_{pid}_{chain_id}_step{step_id} : True := by trivial"
            raw_resp = f"```lean\n{lean_code}\n```"
        else:
            messages = build_initial_messages(system_prompt, filled)
            lean_code = ""
            raw_resp = ""
            success = False
            last_error = ""

            for round_idx in range(args.max_rounds):
                try:
                    raw_resp = call_llm(
                        messages=messages,
                        model=selected_model,
                        mock=False,
                        retries=1,
                        timeout=240,
                        max_tokens=2048 if round_idx == 0 else 4096,
                        reasoning=False
                    )
                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
                    raw_resp = f"```lean\n-- LLM error: {e}\n```"
                    break

                lean_code = extract_lean_block(raw_resp)
                lean_code = post_process_code(lean_code)

                # Attempt compilation (with safe timeout handling)
                ok, output = compile_lean(lean_code, args.project_dir, timeout=120)
                if ok:
                    success = True
                    logger.info(f"Compilation succeeded on round {round_idx+1}")
                    break
                else:
                    last_error = output
                    logger.info(f"Round {round_idx+1} failed, errors:\n{output[:500]}")
                    if round_idx < args.max_rounds - 1:
                        add_feedback(messages, raw_resp, output)

            if not success and not args.mock:
                logger.warning(f"Failed to compile after {args.max_rounds} rounds. Last error:\n{last_error[:500]}")

        fname = f"{pid}__{chain_id}__step{step_id}.lean"
        out_file = out_base / fname
        with out_file.open('w', encoding='utf-8') as fh:
            fh.write(f"-- Generated by generate_lean.py (success={success})\n-- source id: {pid}\n\n")
            fh.write(lean_code + "\n")
        (out_base / (fname + '.resp.txt')).write_text(raw_resp, encoding='utf-8')
        logger.info(f"Generated {out_file} (success={success})")


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


if __name__ == '__main__':
    main()