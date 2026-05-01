from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


STEP_RE = re.compile(
    r"(?ms)(?:^|\n)\s*Step\s+(\d+)\s*:\s*(.*?)(?=(?:\n\s*Step\s+\d+\s*:)|(?:\n\s*Final Answer\s*:)|\Z)"
)
THINK_RE = re.compile(r"(?is)<think>(.*?)(?:</think>|\Z)")
FINAL_RE = re.compile(r"(?is)Final Answer\s*:\s*(.+?)(?:\n\s*\Z|\Z)")
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


@dataclass(frozen=True)
class StepSpan:
    index: int
    text: str
    start_char: int
    end_char: int


def extract_steps(text: str) -> list[StepSpan]:
    steps: list[StepSpan] = []
    for match in STEP_RE.finditer(text):
        full_start = match.start()
        full_end = match.end()
        steps.append(
            StepSpan(
                index=int(match.group(1)),
                text=match.group(2).strip(),
                start_char=full_start,
                end_char=full_end,
            )
        )
    if steps:
        return steps
    return extract_paragraph_steps(text)


def extract_paragraph_steps(text: str) -> list[StepSpan]:
    match = THINK_RE.search(text)
    if match:
        body_start = match.start(1)
        body = match.group(1)
    else:
        body_start = 0
        body = text

    steps: list[StepSpan] = []
    for paragraph_match in re.finditer(r"(?:^|\n\s*\n)([^\S\n]*\S.*?)(?=\n\s*\n|\Z)", body, flags=re.S):
        paragraph = paragraph_match.group(1).strip()
        if not paragraph:
            continue
        if paragraph.lower().startswith("final answer"):
            continue
        start = body_start + paragraph_match.start(1)
        end = body_start + paragraph_match.end(1)
        steps.append(
            StepSpan(
                index=len(steps) + 1,
                text=paragraph,
                start_char=start,
                end_char=end,
            )
        )
    return steps


def extract_final_answer(text: str) -> str | None:
    match = FINAL_RE.search(text)
    if match:
        return match.group(1).strip()
    boxes = BOXED_RE.findall(text)
    if boxes:
        return boxes[-1].strip()
    return None


def extract_reference_answer(solution: str) -> str | None:
    boxes = BOXED_RE.findall(solution)
    if boxes:
        return boxes[-1].strip()
    return extract_final_answer(solution)


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    answer = re.sub(r"^\$|\$$", "", answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = re.sub(r"\s+", "", answer)
    answer = answer.rstrip(".")
    return answer


def rough_answer_match(predicted: str | None, reference: str | None) -> bool | None:
    pred = normalize_answer(predicted)
    ref = normalize_answer(reference)
    if pred is None or ref is None:
        return None
    return pred == ref or pred in ref or ref in pred


def step_end_token_indices(
    tokenizer: Any,
    prompt: str,
    completion: str,
    steps: list[StepSpan],
) -> dict[int, int]:
    full_text = prompt + completion
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    prompt_len = len(prompt)
    result: dict[int, int] = {}
    for step in steps:
        absolute_end = prompt_len + step.end_char
        candidates = [i for i, (_, end) in enumerate(offsets) if end <= absolute_end]
        if not candidates:
            continue
        result[step.index] = candidates[-1]
    return result


def steps_as_dicts(steps: list[StepSpan]) -> list[dict[str, Any]]:
    return [
        {
            "index": step.index,
            "text": step.text,
            "start_char": step.start_char,
            "end_char": step.end_char,
        }
        for step in steps
    ]


def steps_from_dicts(rows: list[dict[str, Any]]) -> list[StepSpan]:
    return [
        StepSpan(
            index=int(row["index"]),
            text=str(row["text"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
        )
        for row in rows
    ]
