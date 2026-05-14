from __future__ import annotations

from fhis.steps import (
    canonical_answer,
    extract_final_answer,
    extract_reference_answer,
    extract_steps,
    rough_answer_match,
)


def test_extract_steps_and_final_answer() -> None:
    text = """<think>
Step 1: Let x=2. This follows from the equation.
Step 2: Substitute x. We get 4.
</think>
Final Answer: 4
"""
    steps = extract_steps(text)
    assert [step.index for step in steps] == [1, 2]
    assert "Let x=2" in steps[0].text
    assert extract_final_answer(text) == "4"


def test_reference_boxed_answer() -> None:
    assert extract_reference_answer(r"The answer is \boxed{42}.") == "42"
    assert rough_answer_match(" 42.", r"42")
    assert rough_answer_match(r"\[\boxed{\frac{1}{2}}\]", r"\frac{1}{2}")
    assert not rough_answer_match(r"\frac{3}{2}", "3")


def test_latex_answer_normalization_handles_common_equivalent_forms() -> None:
    assert canonical_answer(r"\[\boxed{2^k}\]") == canonical_answer(r"$2^{k}$")
    assert rough_answer_match(r"\(\boxed{\dfrac{45}{2}}\).", "22.5")
    assert rough_answer_match(r"Final Answer: \[\boxed{2^{k}}\]", r"$2^k$")
    assert rough_answer_match(r"\frac{2016}{2017^{2}}", r"\dfrac{2016}{2017^2}")


def test_extract_paragraph_steps_fallback() -> None:
    text = """<think>
First compute the discriminant.

Then solve the resulting quadratic equation.

Final Answer: 2
</think>
"""
    steps = extract_steps(text)
    assert [step.index for step in steps] == [1, 2]
    assert "discriminant" in steps[0].text
    assert "quadratic" in steps[1].text


def test_paragraph_fallback_only_inside_think() -> None:
    text = """First compute the discriminant.

Then solve the resulting quadratic equation.

Final Answer: 2
"""
    assert extract_steps(text) == []
