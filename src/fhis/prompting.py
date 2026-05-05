from __future__ import annotations

from typing import Any


STEP_PROMPT = """Solve the problem carefully.

Write your visible solution using this exact format.
Start your answer with "Step 1:".
Every reasoning paragraph must begin with "Step k:" where k is the step number.
Each step should contain one main mathematical claim or computation.
Do not merge multiple independent derivations into one step.
Do not use bullet-only reasoning.
Do not use markdown headings like "**Step 1:**".
Do not write an introduction before Step 1.

Step 1: [claim]. [reasoning/computation].
Step 2: [claim]. [reasoning/computation].
Step 3: [claim]. [reasoning/computation].
...

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
