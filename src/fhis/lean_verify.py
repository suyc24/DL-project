from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LeanVerificationResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    lean_file: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PLACEHOLDER_RE = re.compile(r"\b(?:sorry|admit)\b")
TACTIC_REPLACEMENTS = (
    "native_decide",
    "omega",
    "norm_num",
    "ring_nf",
    "ring",
    "nlinarith",
    "linarith",
)


def lean_command(workdir: str | Path | None = None, executable: str = "lean") -> list[str]:
    root = Path(workdir or ".")
    if (root / "lakefile.lean").exists() or (root / "lakefile.toml").exists():
        return ["lake", "env", executable]
    return [executable]


def candidate_codes(code: str) -> list[tuple[str, str | None]]:
    if re.search(r"\baxiom\b", code):
        return []
    if not PLACEHOLDER_RE.search(code):
        return [(code, None)]
    return [(PLACEHOLDER_RE.sub(tactic, code), tactic) for tactic in TACTIC_REPLACEMENTS]


def ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_lean_candidate(
    code: str,
    workdir: str | Path | None,
    executable: str,
    timeout_s: float,
    keep_file: bool,
) -> LeanVerificationResult:
    root = Path(workdir) if workdir else None
    if root is not None and not root.exists():
        return LeanVerificationResult(
            status="formalization_failed",
            returncode=None,
            stdout="",
            stderr=f"Lean workdir does not exist: {root}",
        )

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".lean",
        delete=not keep_file,
        dir=str(root) if root else None,
    ) as f:
        f.write(code)
        f.write("\n")
        f.flush()
        path = Path(f.name)
        try:
            completed = subprocess.run(
                [*lean_command(workdir=root, executable=executable), str(path)],
                cwd=root,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except FileNotFoundError as exc:
            return LeanVerificationResult(
                status="formalization_failed",
                returncode=None,
                stdout="",
                stderr=f"Lean executable not found: {exc}",
                lean_file=str(path) if keep_file else None,
            )
        except subprocess.TimeoutExpired as exc:
            return LeanVerificationResult(
                status="formalization_failed",
                returncode=None,
                stdout=ensure_text(exc.stdout),
                stderr=ensure_text(exc.stderr) or "Lean verification timed out",
                lean_file=str(path) if keep_file else None,
            )

        return LeanVerificationResult(
            status="proved" if completed.returncode == 0 else "failed",
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            lean_file=str(path) if keep_file else None,
        )


def verify_lean_code(
    code: str,
    workdir: str | Path | None = None,
    executable: str = "lean",
    timeout_s: float = 30.0,
    keep_file: bool = False,
) -> LeanVerificationResult:
    code = code.replace("\x00", "").strip()
    if not code:
        return LeanVerificationResult(
            status="formalization_failed",
            returncode=None,
            stdout="",
            stderr="empty Lean code",
        )
    if "formalization_failed" in code.lower():
        return LeanVerificationResult(
            status="formalization_failed",
            returncode=None,
            stdout="",
            stderr="formalizer reported formalization_failed",
        )

    candidates = candidate_codes(code)
    if not candidates:
        return LeanVerificationResult(
            status="formalization_failed",
            returncode=None,
            stdout="",
            stderr="Lean code contains axiom or an unfinished proof placeholder",
        )

    last_result: LeanVerificationResult | None = None
    for candidate, tactic in candidates:
        if PLACEHOLDER_RE.search(candidate):
            last_result = LeanVerificationResult(
                status="formalization_failed",
                returncode=None,
                stdout="",
                stderr="Lean code still contains an unfinished proof placeholder",
            )
            continue

        result = _run_lean_candidate(
            candidate,
            workdir=workdir,
            executable=executable,
            timeout_s=timeout_s,
            keep_file=keep_file,
        )
        if result.status == "proved":
            if tactic is None:
                return result
            return LeanVerificationResult(
                status="proved",
                returncode=result.returncode,
                stdout=f"placeholder filled with tactic: {tactic}\n{result.stdout}",
                stderr=result.stderr,
                lean_file=result.lean_file,
            )
        if tactic is not None and result.status == "failed":
            result = LeanVerificationResult(
                status="formalization_failed",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                lean_file=result.lean_file,
            )
        last_result = result

    assert last_result is not None
    return last_result
