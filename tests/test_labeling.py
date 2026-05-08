from __future__ import annotations

from fhis.labeling import (
    build_label_prompt,
    fhis_step_label,
    is_labeling_candidate,
    label_is_structurally_valid,
    normalize_label,
)


def sample_trace() -> dict:
    return {
        "trace_id": "trace-1",
        "problem_id": "problem-1",
        "problem": "Compute 1+1.",
        "reference_answer": ["2"],
        "reference_solution": "It is 2.",
        "final_answer": r"\boxed{2}",
        "rough_final_correct": True,
        "steps": [
            {"index": 1, "text": "Compute the sum."},
            {"index": 2, "text": "Conclude the answer is 2."},
        ],
    }


def test_build_label_prompt_contains_canonical_context() -> None:
    prompt = build_label_prompt(sample_trace())

    assert "high-quality FHIS labels" in prompt
    assert "Rough automatic final-answer match:\ntrue" in prompt
    assert "Reference solution:\nIt is 2." in prompt
    assert "Step 1: Compute the sum." in prompt
    assert '"first_invalid_step": integer or null' in prompt


def test_normalize_label_adds_trace_metadata() -> None:
    label = normalize_label(
        {
            "final_correct": "true",
            "first_invalid_step": "null",
            "error_type": None,
            "reason": " correct ",
            "confidence": "HIGH",
        },
        trace=sample_trace(),
        labeler="local_codex",
        labeler_model="gpt-5.5",
        labeler_reasoning_effort="high",
    )

    assert label["trace_id"] == "trace-1"
    assert label["final_correct"] is True
    assert label["problem_id"] == "problem-1"
    assert label["first_invalid_step"] is None
    assert label["reason"] == "correct"
    assert label["confidence"] == "high"
    assert label["rough_final_correct"] is True
    assert label["num_steps"] == 2
    assert label["labeler"] == "local_codex"
    assert label["labeler_model"] == "gpt-5.5"
    assert label["labeler_reasoning_effort"] == "high"


def test_label_validation_uses_num_steps_when_available() -> None:
    label = normalize_label(
        {
            "final_correct": "false",
            "first_invalid_step": 3,
            "error_type": "algebra",
            "reason": "bad step",
            "confidence": "high",
        },
        trace=sample_trace(),
    )

    assert label["final_correct"] is False
    assert not label_is_structurally_valid(label)
    label["first_invalid_step"] = 2
    assert label_is_structurally_valid(label)
    label["reason"] = ""
    assert not label_is_structurally_valid(label)


def test_labeling_candidate_filters_unknown_by_default() -> None:
    trace = sample_trace()
    assert is_labeling_candidate(trace)

    trace["final_answer"] = None
    trace["rough_final_correct"] = None
    assert not is_labeling_candidate(trace)
    assert is_labeling_candidate(trace, include_unknown=True)


def test_fhis_step_label_keeps_invalid_step_when_final_answer_recovers() -> None:
    label = normalize_label(
        {
            "final_correct": True,
            "first_invalid_step": 2,
            "error_type": "bad intermediate claim",
            "reason": "invalid but recovered",
            "confidence": "high",
        },
        trace=sample_trace(),
    )

    assert fhis_step_label(label, 1) == 0
    assert fhis_step_label(label, 2) == 1
