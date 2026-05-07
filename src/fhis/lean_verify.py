from __future__ import annotations

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


def lean_command(workdir: str | Path | None = None, executable: str = "lean") -> list[str]:
    root = Path(workdir or ".")
    if (root / "lakefile.lean").exists() or (root / "lakefile.toml").exists():
        return ["lake", "env", executable]
    return [executable]


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

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".lean",
        delete=not keep_file,
        dir=workdir if workdir else None,
    ) as f:
        f.write(code)
        f.write("\n")
        f.flush()
        path = Path(f.name)
        try:
            completed = subprocess.run(
                [*lean_command(workdir=workdir, executable=executable), str(path)],
                cwd=workdir,
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
                stdout=exc.stdout or "",
                stderr=exc.stderr or "Lean verification timed out",
                lean_file=str(path) if keep_file else None,
            )

        status = "proved" if completed.returncode == 0 else "failed"
        return LeanVerificationResult(
            status=status,
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            lean_file=str(path) if keep_file else None,
        )
