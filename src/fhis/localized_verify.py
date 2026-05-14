from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from fhis.lean_verify import LeanVerificationResult, verify_lean_code


@dataclass(frozen=True)
class ParsedExpression:
    value: Fraction
    lean: str


@dataclass(frozen=True)
class AtomicClaim:
    text: str
    relation: str
    left: str
    right: str
    lean_code: str
    expected_truth: bool
    kind: str = "arithmetic"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckedAtomicClaim:
    claim: AtomicClaim
    status: str
    verification: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status,
            "verification": self.verification,
        }


@dataclass(frozen=True)
class LocalizedVerificationResult:
    status: str
    claims: list[CheckedAtomicClaim]
    lean_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "claims": [claim.to_dict() for claim in self.claims],
            "lean_code": self.lean_code,
        }


class ExpressionParseError(ValueError):
    pass


DECIMAL_RE = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")
RELATION_RE = re.compile(r"<=|>=|~=|=|<|>")
SQRT_APPROX_RE = re.compile(
    r"sqrt\(\s*(?P<radicand>-?\d+(?:\.\d+)?)\s*\)\s*~=\s*(?P<approx>-?\d+(?:\.\d+)?)"
)
ALLOWED_EXPR_CHARS = set("0123456789.+-*/^() ")
CONTRADICTION_RE = re.compile(
    r"\b(?:contradiction|impossible|no solution|not a solution|cannot|not an integer|"
    r"not a perfect square)\b",
    flags=re.I,
)
EUCLIDEAN_DIVISION_CONTEXT_RE = re.compile(
    r"\b(?:remainder|quotient|integer division|modulo|mod)\b",
    flags=re.I,
)
APPROXIMATE_CONTEXT_RE = re.compile(
    r"\b(?:approximately|approx\.?|about|roughly|around)\b",
    flags=re.I,
)


def replace_latex_command_args(text: str, command: str, formatter) -> str:
    marker = "\\" + command
    start = 0
    pieces: list[str] = []
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            pieces.append(text[start:])
            return "".join(pieces)
        pieces.append(text[start:idx])
        pos = idx + len(marker)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            pieces.append(text[idx:pos])
            start = pos
            continue
        args: list[str] = []
        ok = True
        for _ in range(2 if command == "frac" else 1):
            if pos >= len(text) or text[pos] != "{":
                ok = False
                break
            depth = 0
            arg_start = pos + 1
            pos += 1
            while pos < len(text):
                char = text[pos]
                if char == "{":
                    depth += 1
                elif char == "}":
                    if depth == 0:
                        args.append(text[arg_start:pos])
                        pos += 1
                        break
                    depth -= 1
                pos += 1
            else:
                ok = False
                break
            while pos < len(text) and text[pos].isspace():
                pos += 1
        if not ok:
            pieces.append(text[idx:pos])
            start = pos
            continue
        pieces.append(formatter(*args))
        start = pos


def normalize_math_text(text: str) -> str:
    text = text.replace("\u2212", "-").replace("\u00d7", "*")
    text = text.replace("\u2264", "<=").replace("\u2265", ">=").replace("\u2248", "~=")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = replace_latex_command_args(
        text,
        "frac",
        lambda numerator, denominator: f"(({normalize_math_text(numerator)})/({normalize_math_text(denominator)}))",
    )
    text = replace_latex_command_args(
        text,
        "sqrt",
        lambda radicand: f"sqrt({normalize_math_text(radicand)})",
    )
    replacements = {
        r"\cdot": "*",
        r"\times": "*",
        r"\div": "/",
        r"\leqslant": "<=",
        r"\geqslant": ">=",
        r"\leq": "<=",
        r"\geq": ">=",
        r"\le": "<=",
        r"\ge": ">=",
        r"\approx": "~=",
        r"\sim": "~=",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", text)
    text = text.replace("\\[", " ").replace("\\]", " ")
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("$", " ")
    text = text.replace("{", "(").replace("}", ")")
    return text


def _wrap_decimal_literals(expr: str) -> str:
    return DECIMAL_RE.sub(lambda match: f'F("{match.group(0)}")', expr)


def _prepare_expr(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<=\d)\s+(?=\d)", "", expr)
    expr = re.sub(r"(?<=[\d)])\s*\(", "*(", expr)
    expr = re.sub(r"\)\s*(?=\d)", ")*", expr)
    expr = _wrap_decimal_literals(expr)
    return expr


def fraction_from_decimal_string(text: str) -> Fraction:
    try:
        return Fraction(text)
    except ValueError as exc:
        raise ExpressionParseError(f"invalid decimal literal: {text}") from exc


def lean_rat(value: Fraction) -> str:
    if value.denominator == 1:
        return f"({value.numerator} : Rat)"
    return f"(({value.numerator} : Rat) / ({value.denominator} : Rat))"


def parse_numeric_expression(expr: str) -> ParsedExpression:
    prepared = _prepare_expr(expr)
    if not prepared or any(name in prepared for name in ("sqrt", "log", "sin", "cos")):
        raise ExpressionParseError(f"unsupported expression: {expr}")
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise ExpressionParseError(f"could not parse expression: {expr}") from exc
    return _eval_ast(tree.body)


def _eval_ast(node: ast.AST) -> ParsedExpression:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ExpressionParseError("booleans are not arithmetic expressions")
        if isinstance(node.value, int):
            value = Fraction(node.value)
            return ParsedExpression(value=value, lean=lean_rat(value))
        raise ExpressionParseError(f"unsupported literal: {node.value!r}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "F" or len(node.args) != 1:
            raise ExpressionParseError("unsupported function call")
        arg = node.args[0]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            raise ExpressionParseError("decimal wrapper requires a string literal")
        value = fraction_from_decimal_string(arg.value)
        return ParsedExpression(value=value, lean=lean_rat(value))

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return inner
        return ParsedExpression(value=-inner.value, lean=f"(-{inner.lean})")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return ParsedExpression(left.value + right.value, f"({left.lean} + {right.lean})")
        if isinstance(node.op, ast.Sub):
            return ParsedExpression(left.value - right.value, f"({left.lean} - {right.lean})")
        if isinstance(node.op, ast.Mult):
            return ParsedExpression(left.value * right.value, f"({left.lean} * {right.lean})")
        if isinstance(node.op, ast.Div):
            if right.value == 0:
                raise ExpressionParseError("division by zero")
            return ParsedExpression(left.value / right.value, f"({left.lean} / {right.lean})")
        if isinstance(node.op, ast.Pow):
            if right.value.denominator != 1:
                raise ExpressionParseError("fractional powers are not supported")
            exponent = right.value.numerator
            if exponent < 0 or exponent > 24:
                raise ExpressionParseError("unsupported exponent")
            return ParsedExpression(left.value**exponent, f"({left.lean} ^ {exponent})")
    raise ExpressionParseError(f"unsupported expression node: {type(node).__name__}")


def compare_values(left: Fraction, relation: str, right: Fraction, tolerance: Fraction | None = None) -> bool:
    if relation == "=":
        return left == right
    if relation == "<":
        return left < right
    if relation == ">":
        return left > right
    if relation == "<=":
        return left <= right
    if relation == ">=":
        return left >= right
    if relation == "~=":
        tol = tolerance if tolerance is not None else Fraction(1, 100)
        return abs(left - right) <= tol
    raise ValueError(f"unsupported relation: {relation}")


def relation_to_lean(left: str, relation: str, right: str) -> str:
    if relation == "=":
        return f"{left} = {right}"
    if relation == "~=" :
        raise ValueError("approximate relations need an explicit tolerance")
    return f"{left} {relation} {right}"


def decimal_precision(text: str) -> int:
    match = re.search(r"-?\d+\.(\d+)", text)
    return len(match.group(1)) if match else 0


def tolerance_from_text(text: str) -> Fraction:
    precision = decimal_precision(text)
    if precision <= 0:
        return Fraction(1, 2)
    return Fraction(1, 2 * (10**precision))


def strip_side(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+([.,;:])$", r"\1", text)
    text = text.strip(",;:")
    if text.endswith(".") and (text.count(".") != 1 or re.fullmatch(r".*\d\.", text)):
        text = text[:-1].rstrip()
    while text.startswith(("+", "*", "/", "^")):
        text = text[1:].lstrip()
    while text.endswith(("+", "-", "*", "/", "^")):
        text = text[:-1].rstrip()
    return text


def expand_left_span(text: str, relation_start: int) -> tuple[str, int, int]:
    pos = relation_start - 1
    while pos >= 0 and text[pos] in ALLOWED_EXPR_CHARS:
        pos -= 1
    raw_start = pos + 1
    raw = text[raw_start:relation_start]
    stripped = strip_side(raw)
    offset = raw.find(stripped) if stripped else 0
    start = raw_start + max(offset, 0)
    return stripped, start, start + len(stripped)


def expand_right_span(text: str, relation_end: int) -> tuple[str, int, int]:
    pos = relation_end
    while pos < len(text) and text[pos] in ALLOWED_EXPR_CHARS:
        pos += 1
    raw = text[relation_end:pos]
    stripped = strip_side(raw)
    offset = raw.find(stripped) if stripped else 0
    start = relation_end + max(offset, 0)
    return stripped, start, start + len(stripped)


def clean_numeric_boundaries(text: str, left_start: int, right_end: int, left_text: str) -> bool:
    if left_start > 0:
        before = text[left_start - 1]
        if before in {"_", "\\", "^", "!"} or before.isalpha():
            return False

    prefix = text[:left_start]
    prev_match = re.search(r"\S\s*$", prefix)
    prev = prev_match.group(0).strip() if prev_match else ""
    if prev in {"^", "_", "\\", "!"}:
        return False
    if prev in {"+", "-", "*", "/"} and re.fullmatch(r"[+-]?\s*\d+(?:\.\d+)?", left_text):
        return False
    if left_text.startswith(("+", "-")) and prev not in {"", "(", "[", "{", ",", ";", ":"}:
        return False

    prefix_match = re.search(r"([\\A-Za-z][\\A-Za-z0-9_]*)\s*$", prefix)
    if prefix_match:
        token = prefix_match.group(1)
        if "_" in token or "\\" in token or token in {"sqrt", "log", "sin", "cos", "tan"}:
            return False

    suffix = text[right_end:]
    suffix_match = re.match(r"\s*(?:[_\\]|[A-Za-z](?=$|[\s()^_*/+\-]))", suffix)
    if suffix_match:
        return False
    if right_end < len(text):
        after = text[right_end]
        if after in {"_", "\\", "!"} or after.isalpha():
            return False
    return True


def exact_claim_is_contextually_safe(
    text: str,
    left_start: int,
    right_end: int,
    left_text: str,
    relation: str,
    right_text: str,
) -> bool:
    if relation != "=":
        return True

    window = text[max(0, left_start - 80) : min(len(text), right_end + 120)]
    if APPROXIMATE_CONTEXT_RE.search(window):
        return False
    if ("/" in left_text or "/" in right_text) and EUCLIDEAN_DIVISION_CONTEXT_RE.search(window):
        return False
    return True


def make_exact_claim(left_text: str, relation: str, right_text: str) -> AtomicClaim | None:
    try:
        left = parse_numeric_expression(left_text)
        right = parse_numeric_expression(right_text)
    except ExpressionParseError:
        return None
    if relation == "~=":
        tolerance = max(tolerance_from_text(left_text), tolerance_from_text(right_text))
        truth = compare_values(left.value, relation, right.value, tolerance=tolerance)
        lower_prop = relation_to_lean(left.lean, "<=", f"({right.lean} + {lean_rat(tolerance)})")
        upper_prop = relation_to_lean(right.lean, "<=", f"({left.lean} + {lean_rat(tolerance)})")
        prop = f"({lower_prop}) /\\ ({upper_prop})"
    else:
        truth = compare_values(left.value, relation, right.value)
        prop = relation_to_lean(left.lean, relation, right.lean)
    lean_code = f"example : {prop} := by\n  native_decide\n"
    return AtomicClaim(
        text=f"{left_text} {relation} {right_text}",
        relation=relation,
        left=left_text,
        right=right_text,
        lean_code=lean_code,
        expected_truth=truth,
        kind="arithmetic",
    )


def make_sqrt_approx_claim(match: re.Match[str]) -> AtomicClaim | None:
    radicand_text = match.group("radicand")
    approx_text = match.group("approx")
    try:
        radicand = fraction_from_decimal_string(radicand_text)
        approx = fraction_from_decimal_string(approx_text)
    except ExpressionParseError:
        return None
    if radicand < 0 or radicand.denominator != 1 or approx < 0:
        return None
    tolerance = tolerance_from_text(approx_text)
    lower = max(Fraction(0), approx - tolerance)
    upper = approx + tolerance
    truth = lower * lower <= radicand <= upper * upper
    prop = (
        f"(({lean_rat(lower)} ^ 2) <= {lean_rat(radicand)}) /\\ "
        f"({lean_rat(radicand)} <= ({lean_rat(upper)} ^ 2))"
    )
    lean_code = f"example : {prop} := by\n  native_decide\n"
    return AtomicClaim(
        text=f"sqrt({radicand_text}) ~= {approx_text}",
        relation="~=",
        left=f"sqrt({radicand_text})",
        right=approx_text,
        lean_code=lean_code,
        expected_truth=truth,
        kind="sqrt_approx",
    )


def contradiction_context(text: str) -> bool:
    return CONTRADICTION_RE.search(text) is not None


def extract_atomic_claims(step_text: str, max_claims: int = 16) -> list[AtomicClaim]:
    normalized = normalize_math_text(step_text)
    has_contradiction_context = contradiction_context(step_text)
    claims: list[AtomicClaim] = []
    seen: set[tuple[str, str, str, str]] = set()

    sqrt_spans = [match.span() for match in SQRT_APPROX_RE.finditer(normalized)]
    for match in SQRT_APPROX_RE.finditer(normalized):
        claim = make_sqrt_approx_claim(match)
        if claim is None:
            continue
        if has_contradiction_context and not claim.expected_truth:
            continue
        key = (claim.kind, claim.left, claim.relation, claim.right)
        if key not in seen:
            seen.add(key)
            claims.append(claim)

    for match in RELATION_RE.finditer(normalized):
        if any(start <= match.start() < end for start, end in sqrt_spans):
            continue
        relation = match.group(0)
        left_text, left_start, _ = expand_left_span(normalized, match.start())
        right_text, _, right_end = expand_right_span(normalized, match.end())
        if not left_text or not right_text:
            continue
        if not clean_numeric_boundaries(normalized, left_start, right_end, left_text):
            continue
        if not exact_claim_is_contextually_safe(
            normalized,
            left_start,
            right_end,
            left_text,
            relation,
            right_text,
        ):
            continue
        if "sqrt" in left_text or "sqrt" in right_text:
            continue
        claim = make_exact_claim(left_text, relation, right_text)
        if claim is None:
            continue
        if has_contradiction_context and not claim.expected_truth:
            continue
        key = (claim.kind, claim.left, claim.relation, claim.right)
        if key in seen:
            continue
        seen.add(key)
        claims.append(claim)
        if len(claims) >= max_claims:
            break
    return claims[:max_claims]


def build_localized_lean_code(step_text: str, max_claims: int = 16) -> str | None:
    claims = extract_atomic_claims(step_text, max_claims=max_claims)
    if not claims:
        return None
    parts = ["-- Deterministic localized arithmetic checks generated by fhis.localized_verify."]
    for index, claim in enumerate(claims, start=1):
        parts.append(f"-- claim {index}: {claim.text}")
        parts.append(claim.lean_code.rstrip())
    return "\n\n".join(parts) + "\n"


def claim_status_from_lean(claim: AtomicClaim, result: LeanVerificationResult) -> str:
    if result.status == "proved":
        return "proved"
    if result.status == "failed" and not claim.expected_truth:
        return "failed"
    if result.status == "failed":
        return "formalization_failed"
    return result.status


def verify_localized_step(
    step_text: str,
    workdir: str | Path | None = None,
    executable: str = "lean",
    timeout_s: float = 10.0,
    keep_files: bool = False,
    max_claims: int = 16,
) -> LocalizedVerificationResult:
    claims = extract_atomic_claims(step_text, max_claims=max_claims)
    if not claims:
        return LocalizedVerificationResult(status="not_applicable", claims=[], lean_code=None)

    checked: list[CheckedAtomicClaim] = []
    for claim in claims:
        result = verify_lean_code(
            claim.lean_code,
            workdir=workdir,
            executable=executable,
            timeout_s=timeout_s,
            keep_file=keep_files,
        )
        checked.append(
            CheckedAtomicClaim(
                claim=claim,
                status=claim_status_from_lean(claim, result),
                verification=result.to_dict(),
            )
        )

    statuses = [item.status for item in checked]
    if any(status == "failed" for status in statuses):
        status = "failed"
    elif all(status == "proved" for status in statuses):
        status = "proved"
    else:
        status = "formalization_failed"
    return LocalizedVerificationResult(
        status=status,
        claims=checked,
        lean_code=build_localized_lean_code(step_text, max_claims=max_claims),
    )
