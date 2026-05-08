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


def compact_steps(trace: dict[str, Any]) -> str:
    lines = []
    for step in trace.get("steps", []):
        text = str(step.get("text", "")).strip()
        lines.append(f"Step {step['index']}: {text}")
    return "\n\n".join(lines)


def rough_final_correct_text(trace: dict[str, Any]) -> str:
    rough = trace.get("rough_final_correct")
    return "unknown" if rough is None else str(bool(rough)).lower()


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def build_label_prompt(trace: dict[str, Any]) -> str:
    """Build the canonical FHIS annotation prompt used by all labelers."""
    return f"""You are creating high-quality FHIS labels for a mathematical reasoning probe.

Definitions:
- A harmful invalid step is the earliest generated step whose mathematical claim,
  transformation, computation, or conclusion is wrong and can plausibly cause the
  final answer to be wrong.
- final_correct records whether the generated final answer matches the reference.
- first_invalid_step records the earliest harmful invalid step, even if the trace
  later recovers and reaches the correct final answer.
- If the final answer is correct and the generated reasoning has no harmful invalid
  step, set first_invalid_step=null.
- If the final answer is wrong, first_invalid_step should be the earliest harmful
  invalid step. Do not choose a later step if an earlier harmful error exists.
- If the generated trace is incomplete, lacks enough information, or the first
  harmful invalid step cannot be determined, use confidence="low".
- Minor wording issues, missing rigor, or skipped algebra are not harmful invalid
  steps unless they introduce a false claim.

Return only JSON matching this schema:
{{
  "final_correct": true or false,
  "first_invalid_step": integer or null,
  "error_type": string or null,
  "reason": string,
  "confidence": "high" or "medium" or "low"
}}

Trace id:
{trace.get("trace_id")}

Rough automatic final-answer match:
{rough_final_correct_text(trace)}

Problem:
{trace["problem"]}

Reference answer:
{trace.get("reference_answer")}

Reference solution:
{trace.get("reference_solution")}

Generated final answer:
{trace.get("final_answer")}

Generated steps:
{compact_steps(trace)}
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


def normalize_label(
    raw: dict[str, Any],
    trace: dict[str, Any] | None = None,
    trace_id: str | None = None,
    labeler: str | None = None,
    labeler_model: str | None = None,
    labeler_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Normalize a raw annotator response into the canonical FHIS label row."""
    first_invalid = raw.get("first_invalid_step")
    if first_invalid in ("", "null", "None"):
        first_invalid = None
    if first_invalid is not None:
        first_invalid = int(first_invalid)
    if trace is not None:
        trace_id = str(trace["trace_id"])
    if trace_id is None:
        raise ValueError("normalize_label requires trace or trace_id")

    label = {
        "trace_id": str(trace_id),
        "final_correct": coerce_bool(raw.get("final_correct", False)),
        "first_invalid_step": first_invalid,
        "error_type": raw.get("error_type"),
        "reason": str(raw.get("reason", "")).strip(),
        "confidence": str(raw.get("confidence", "low")).lower(),
    }
    if trace is not None:
        if "problem_id" in trace:
            label["problem_id"] = str(trace["problem_id"])
        label["rough_final_correct"] = trace.get("rough_final_correct")
        label["num_steps"] = len(trace.get("steps", []))
    if labeler is not None:
        label["labeler"] = labeler
    if labeler_model is not None:
        label["labeler_model"] = labeler_model
    if labeler_reasoning_effort is not None:
        label["labeler_reasoning_effort"] = labeler_reasoning_effort
    return label


def label_is_structurally_valid(label: dict[str, Any]) -> bool:
    if not str(label.get("reason", "")).strip():
        return False
    if label.get("confidence") not in {"high", "medium", "low"}:
        return False
    first_invalid = label.get("first_invalid_step")
    if first_invalid is None:
        return True
    num_steps = label.get("num_steps")
    if num_steps is None:
        return int(first_invalid) >= 1
    return 1 <= int(first_invalid) <= int(num_steps)


def is_labeling_candidate(trace: dict[str, Any], include_unknown: bool = False) -> bool:
    if not trace.get("steps"):
        return False
    if trace.get("rough_final_correct") is None and not include_unknown:
        return False
    if trace.get("final_answer") is None and not include_unknown:
        return False
    return True


def fhis_step_label(
    label: dict[str, Any],
    step_index: int,
    keep_confidence: str = "high",
) -> int | None:
    if str(label.get("confidence", "")).lower() != keep_confidence:
        return None
    first_invalid = label.get("first_invalid_step")
    if bool(label.get("final_correct", False)) and first_invalid is None:
        return 0
    if first_invalid is None:
        return None
    first_invalid = int(first_invalid)
    if step_index < first_invalid:
        return 0
    if step_index == first_invalid:
        return 1
    return None
