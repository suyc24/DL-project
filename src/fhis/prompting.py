from __future__ import annotations

from typing import Any


STEP_PROMPT = """Solve the problem carefully.

Write your reasoning as clear paragraphs separated by blank lines.

Then output:

Final Answer: [answer]

Problem:
{problem}
"""


def build_user_prompt(problem: str) -> str:
    return STEP_PROMPT.format(problem=problem.strip())


def apply_qwen_chat_template(tokenizer: Any, user_prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
