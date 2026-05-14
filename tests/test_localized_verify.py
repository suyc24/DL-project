from __future__ import annotations

import shutil

import pytest

from fhis.formalizer import FormalizationRequest, build_formalizer
from fhis.lean_verify import verify_lean_code
from fhis.localized_verify import (
    build_localized_lean_code,
    extract_atomic_claims,
    verify_localized_step,
)


def test_extracts_exact_arithmetic_and_sqrt_approximation_claims() -> None:
    claims = extract_atomic_claims(
        r"Since \( \sqrt{184} \approx 13.56 \), and \(15^2 = 225 \geq 184\)."
    )

    assert [(claim.text, claim.expected_truth) for claim in claims] == [
        ("sqrt(184) ~= 13.56", True),
        ("15^2 = 225", True),
        ("225 >= 184", True),
    ]


def test_extracts_false_arithmetic_claim_without_symbolic_noise() -> None:
    claims = extract_atomic_claims(r"We compute \(1 + 1 = 3\), then solve \(n = 9\).")

    assert [(claim.text, claim.expected_truth) for claim in claims] == [("1 + 1 = 3", False)]


def test_skips_division_equalities_in_remainder_context() -> None:
    claims = extract_atomic_claims(
        r"Since \(55 \div 10 = 5\) complete cycles with a remainder of 5 units."
    )

    assert claims == []


def test_does_not_cut_factorial_multiplication_as_numeric_rhs() -> None:
    claims = extract_atomic_claims(
        r"\[15! - 13! = 13!(15 \cdot 14 - 1) = 13! \cdot 209.\]"
    )

    assert claims == []


def test_skips_exact_equalities_in_approximate_context() -> None:
    claims = extract_atomic_claims(
        r"Each group has approximately \(\frac{2009}{3} = 669\) triangles."
    )

    assert claims == []


def test_does_not_treat_subscripts_or_function_arguments_as_numeric_claims() -> None:
    claims = extract_atomic_claims(
        r"We have \(a_2 = 1\) and \(\log_T(2^3) = 3\log_T 2\), while \(8 = 2^3\)."
    )

    assert [(claim.text, claim.expected_truth) for claim in claims] == [("8 = 2^3", True)]


def test_skips_false_atoms_used_as_contradiction_witnesses() -> None:
    claims = extract_atomic_claims(r"This simplifies to \(0 = -44\). This is a contradiction, so no solution.")

    assert claims == []


def test_does_not_cut_symbolic_implicit_multiplication_rhs() -> None:
    claims = extract_atomic_claims(r"Cross-multiplying gives \(512 = 2 f(2)^2\), then \(256 = f(2)^2\).")

    assert claims == []


def test_localized_formalizer_returns_formalization_failed_when_no_supported_claim() -> None:
    formalizer = build_formalizer({"formalizer": {"backend": "localized"}})
    code = formalizer.formalize(
        FormalizationRequest(
            problem="Find n.",
            previous_steps=[],
            current_step_index=1,
            current_step="Let n be the desired integer.",
        )
    )

    assert "formalization_failed" in code


def test_localized_formalizer_generates_native_decide_checks() -> None:
    formalizer = build_formalizer({"formalizer": {"backend": "localized"}})
    code = formalizer.formalize(
        FormalizationRequest(
            problem="Compute 1 + 1.",
            previous_steps=[],
            current_step_index=1,
            current_step="We compute 1 + 1 = 2.",
        )
    )

    assert "native_decide" in code
    assert "1 + 1" in code


@pytest.mark.skipif(shutil.which("lean") is None, reason="Lean executable is not installed")
def test_verify_localized_step_proves_true_and_refutes_false_claims() -> None:
    proved = verify_localized_step("We compute 1 + 1 = 2.", timeout_s=10.0)
    failed = verify_localized_step("We compute 1 + 1 = 3.", timeout_s=10.0)

    assert proved.status == "proved"
    assert failed.status == "failed"


@pytest.mark.skipif(shutil.which("lean") is None, reason="Lean executable is not installed")
def test_localized_formalizer_false_claim_code_is_a_lean_veto() -> None:
    code = build_localized_lean_code("We compute 1 + 1 = 3.")
    assert code is not None

    result = verify_lean_code(code, timeout_s=10.0)

    assert result.status == "failed"


@pytest.mark.skipif(shutil.which("lean") is None, reason="Lean executable is not installed")
def test_verify_lean_code_tries_safe_tactics_for_placeholders() -> None:
    proved = verify_lean_code("example : 1 + 1 = 2 := by sorry", timeout_s=10.0)
    not_proved = verify_lean_code("example : 1 + 1 = 3 := by sorry", timeout_s=10.0)

    assert proved.status == "proved"
    assert "placeholder filled" in proved.stdout
    assert not_proved.status == "formalization_failed"
