from __future__ import annotations

import json
import re
from typing import Any


LABEL_SCHEMA = """Return only valid JSON with this schema:
{
  "final_correct": true or false,
  "first_invalid_step": integer or null,
  "error_type": string or null,
  "reason": string,
  "confidence": "high" or "medium" or "low"
}
"""


def build_label_prompt(trace: dict[str, Any]) -> str:
    steps_text = "\n".join(
        f"Step {step['index']}: {step['text']}" for step in trace.get("steps", [])
    )
    return f"""You are labeling mathematical reasoning traces for a probing experiment.

Task:
1. Decide whether the generated final answer is correct.
2. If the final answer is wrong, identify the first harmful invalid step: the earliest step
   whose mathematical claim or transformation is invalid and can plausibly cause the wrong answer.
3. If the final answer is correct and all reasoning is valid, set first_invalid_step to null.
4. If unsure, use confidence medium or low.

Problem:
{trace["problem"]}

Reference answer:
{trace.get("reference_answer")}

Reference solution:
{trace.get("reference_solution")}

Generated final answer:
{trace.get("final_answer")}

Generated steps:
{steps_text}

{LABEL_SCHEMA}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_label(raw: dict[str, Any], trace_id: str) -> dict[str, Any]:
    first_invalid = raw.get("first_invalid_step")
    if first_invalid in ("", "null", "None"):
        first_invalid = None
    if first_invalid is not None:
        first_invalid = int(first_invalid)
    return {
        "trace_id": trace_id,
        "final_correct": bool(raw.get("final_correct", False)),
        "first_invalid_step": first_invalid,
        "error_type": raw.get("error_type"),
        "reason": str(raw.get("reason", "")),
        "confidence": str(raw.get("confidence", "low")).lower(),
    }


def fhis_step_label(
    label: dict[str, Any],
    step_index: int,
    keep_confidence: str = "high",
) -> int | None:
    if str(label.get("confidence", "")).lower() != keep_confidence:
        return None
    if bool(label.get("final_correct", False)):
        return 0
    first_invalid = label.get("first_invalid_step")
    if first_invalid is None:
        return None
    first_invalid = int(first_invalid)
    if step_index < first_invalid:
        return 0
    if step_index == first_invalid:
        return 1
    return None
