from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


STEP_RE = re.compile(
    r"(?ms)(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?Step\s+(\d+)\s*(?::|\*\*:)\s*"
    r"(.*?)(?=(?:\n\s*(?:#{1,6}\s*)?(?:\*\*)?Step\s+\d+\s*(?::|\*\*:))|"
    r"(?:\n\s*Final Answer\s*:)|\Z)"
)
NUMBERED_STEP_RE = re.compile(
    r"(?ms)(?:^|\n)\s*(\d+)[.)]\s+(.*?)(?=(?:\n\s*\d+[.)]\s+)|(?:\n\s*Final Answer\s*:)|\Z)"
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
                index=len(steps) + 1,
                text=match.group(2).strip(),
                start_char=full_start,
                end_char=full_end,
            )
        )
    if steps:
        return steps
    if THINK_RE.search(text):
        return extract_paragraph_steps(text)
    return extract_numbered_steps(text)


def extract_numbered_steps(text: str) -> list[StepSpan]:
    steps: list[StepSpan] = []
    for match in NUMBERED_STEP_RE.finditer(text):
        content = match.group(2).strip()
        if not content:
            continue
        steps.append(
            StepSpan(
                index=len(steps) + 1,
                text=content,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return steps


def extract_paragraph_steps(text: str) -> list[StepSpan]:
    match = THINK_RE.search(text)
    if not match:
        return []
    body_start = match.start(1)
    body = match.group(1)

    steps: list[StepSpan] = []
    for paragraph_match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", body, flags=re.S):
        paragraph = paragraph_match.group(0).strip()
        if not paragraph:
            continue
        if paragraph.lower().startswith("final answer"):
            continue
        start = body_start + paragraph_match.start()
        end = body_start + paragraph_match.end()
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
    boxes = extract_latex_command_args(text, "boxed")
    if boxes:
        return boxes[-1].strip()
    return None


def extract_reference_answer(solution: str) -> str | None:
    boxes = extract_latex_command_args(solution, "boxed")
    if boxes:
        return boxes[-1].strip()
    return extract_final_answer(solution)


def extract_latex_command_args(text: str, command: str) -> list[str]:
    """Extract braced arguments for a LaTeX command, allowing nested braces."""
    result: list[str] = []
    marker = "\\" + command
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            return result
        pos = idx + len(marker)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            start = pos
            continue
        depth = 0
        arg_start = pos + 1
        pos += 1
        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    result.append(text[arg_start:pos])
                    start = pos + 1
                    break
                depth -= 1
            pos += 1
        else:
            return result


def strip_math_delimiters(answer: str) -> str:
    answer = answer.strip()
    changed = True
    while changed:
        changed = False
        for left, right in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$")):
            if answer.startswith(left) and answer.endswith(right):
                answer = answer[len(left) : len(answer) - len(right)].strip()
                changed = True
    return answer


def _replace_latex_fracs(text: str) -> str:
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    while True:
        idx = text.find(r"\frac")
        if idx < 0:
            return text
        pos = idx + len(r"\frac")
        while pos < len(text) and text[pos].isspace():
            pos += 1
        args: list[tuple[int, int, str]] = []
        for _ in range(2):
            if pos >= len(text) or text[pos] != "{":
                return text
            depth = 0
            arg_start = pos + 1
            pos += 1
            while pos < len(text):
                if text[pos] == "{":
                    depth += 1
                elif text[pos] == "}":
                    if depth == 0:
                        args.append((arg_start - 1, pos + 1, text[arg_start:pos]))
                        pos += 1
                        break
                    depth -= 1
                pos += 1
            else:
                return text
            while pos < len(text) and text[pos].isspace():
                pos += 1
        numerator = _replace_latex_fracs(args[0][2])
        denominator = _replace_latex_fracs(args[1][2])
        replacement = f"(({numerator})/({denominator}))"
        text = text[:idx] + replacement + text[pos:]


def canonical_answer(answer: str | None) -> str | None:
    normalized = normalize_answer(answer)
    if normalized is None:
        return None
    text = normalized
    text = _replace_latex_fracs(text)
    replacements = {
        r"\cdot": "*",
        r"\times": "*",
        r"\div": "/",
        r"\leqslant": r"\le",
        r"\geqslant": r"\ge",
        r"\operatorname": "",
        r"\mathrm": "",
        r"\text": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\(?:,|;|:|!|quad|qquad| )", "", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\{([A-Za-z0-9+\-*/^_=<>.,]+)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("−", "-")
    text = re.sub(r"\s+", "", text)
    text = text.strip(".,;:")
    return text


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    final_match = FINAL_RE.search(answer)
    if final_match:
        answer = final_match.group(1).strip()
    boxed = extract_latex_command_args(answer, "boxed")
    if boxed:
        answer = boxed[-1].strip()
    answer = strip_math_delimiters(answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    answer = re.sub(r"\s+", "", answer)
    answer = answer.strip(".,;:")
    return answer


def simple_numeric_value(answer: str | None) -> Fraction | None:
    normalized = canonical_answer(answer)
    if normalized is None:
        return None
    frac = re.fullmatch(r"\(?\(?(-?\d+)\)?/\(?(-?\d+)\)?\)?", normalized)
    if frac:
        denominator = int(frac.group(2))
        if denominator:
            return Fraction(int(frac.group(1)), denominator)
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        return Fraction(normalized)
    return None


def rough_answer_match(predicted: str | None, reference: str | None) -> bool | None:
    pred = canonical_answer(predicted)
    ref = canonical_answer(reference)
    if pred is None or ref is None:
        return None
    if pred == ref:
        return True
    pred_num = simple_numeric_value(pred)
    ref_num = simple_numeric_value(ref)
    if pred_num is not None and ref_num is not None:
        return pred_num == ref_num
    return False


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
