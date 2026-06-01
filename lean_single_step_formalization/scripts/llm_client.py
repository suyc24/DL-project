#!/usr/bin/env python3
"""Self-contained LLM client for the single-step formalization loop.

Supported providers:

- `openai`: OpenAI-compatible chat-completions API via the `openai` Python
  package and `OPENAI_API_KEY` / `OPENAI_BASE_URL`.
- `codex`: local Codex CLI via `codex exec`; this uses the local Codex login and
  config, not a direct API key from this script.

Both providers expose the same `call_llm` and `call_llm_batch` functions.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _load_env_from_tree(filename: str = ".env", max_levels: int = 6) -> None:
    p = Path(__file__).resolve().parent
    for _ in range(max_levels):
        env_file = p / filename
        if env_file.is_file():
            _parse_dotenv(env_file)
            return
        p = p.parent


def _parse_dotenv(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_from_tree()

logger = logging.getLogger("single_step_llm_client")
if not logger.handlers:
    log_file = os.environ.get("SINGLE_STEP_LLM_LOG_FILE", "single_step_llm_calls.log")
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, os.environ.get("SINGLE_STEP_LLM_LOG_LEVEL", "INFO").upper(), logging.INFO))
    logger.propagate = False

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<{role}>\n{content}\n</{role}>")
    return "\n\n".join(parts)


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if messages is not None:
        return messages
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_openai(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    reasoning: bool | None,
    openai_reasoning_effort: str | None,
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    selected_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o"
    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"), timeout=timeout)
    kwargs: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if openai_reasoning_effort:
        kwargs["reasoning_effort"] = openai_reasoning_effort
    if reasoning is not None:
        kwargs["extra_body"] = {"thinking": {"type": "enabled" if reasoning else "disabled"}}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise RuntimeError("OpenAI-compatible API returned empty content")
    return content


def _call_codex(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    timeout: int,
    reasoning_effort: str,
    sandbox: str,
    cwd: str,
) -> str:
    selected_model = model or os.environ.get("CODEX_MODEL") or os.environ.get("OPENAI_MODEL")
    prompt = _messages_to_prompt(messages).replace("\x00", "")
    env = os.environ.copy()
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".txt") as out:
        cmd = [
            "codex",
            "exec",
            "--cd",
            cwd,
            "--sandbox",
            sandbox,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "-c",
            'approval_policy="never"',
            "--output-last-message",
            out.name,
            "-",
        ]
        if selected_model:
            cmd[cmd.index("-c"):cmd.index("-c")] = ["-m", selected_model]
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            raise TimeoutError(f"codex exec timed out after {timeout}s\n{stderr}")
        if proc.returncode != 0:
            raise RuntimeError(stderr.strip() or stdout.strip() or "codex exec failed")
        out.seek(0)
        content = out.read()
    if not content.strip():
        raise RuntimeError("codex exec returned empty last message")
    return content


def call_llm(
    system_prompt: str = "",
    user_prompt: str = "",
    *,
    messages: list[dict[str, str]] | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    mock: bool = False,
    retries: int = 2,
    backoff: float = 1.0,
    timeout: int = 120,
    reasoning: bool | None = None,
    openai_reasoning_effort: str | None = None,
    codex_reasoning_effort: str = "high",
    codex_sandbox: str = "read-only",
    codex_cwd: str | None = None,
    call_label: str = "",
) -> str:
    """Call an LLM provider and return plain text."""
    label = f"[{call_label}] " if call_label else ""
    selected_provider = provider or os.environ.get("LLM_PROVIDER") or "openai"
    selected_provider = selected_provider.lower()
    built_messages = _build_messages(system_prompt, user_prompt, messages)
    if selected_provider == "codex" and retries == 2:
        retries = 0

    if mock or selected_provider == "mock":
        return (
            "Chain (mock)\n"
            "[Step 1] Mock step 1.\n"
            "[Step 2] Mock step 2.\n"
            "[Final Answer] mock answer\n"
        )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        start = time.time()
        try:
            logger.info(
                "%sprovider=%s model=%s attempt=%d timeout=%s reasoning=%s openai_reasoning_effort=%s",
                label,
                selected_provider,
                model or "default",
                attempt + 1,
                timeout,
                reasoning,
                openai_reasoning_effort,
            )
            if selected_provider == "openai":
                text = _call_openai(
                    built_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    reasoning=reasoning,
                    openai_reasoning_effort=openai_reasoning_effort,
                )
            elif selected_provider == "codex":
                text = _call_codex(
                    built_messages,
                    model=model,
                    timeout=timeout,
                    reasoning_effort=codex_reasoning_effort,
                    sandbox=codex_sandbox,
                    cwd=codex_cwd or str(Path.cwd()),
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {selected_provider}")
            logger.info("%ssucceeded in %.2fs, chars=%d", label, time.time() - start, len(text))
            return text
        except Exception as exc:
            last_error = exc
            logger.warning("%sattempt %d failed after %.2fs: %s", label, attempt + 1, time.time() - start, exc)
            if attempt < retries:
                time.sleep(backoff * (2**attempt))

    return (
        "```lean\n"
        "-- LLM call failed after retries\n"
        f"-- provider: {selected_provider}\n"
        f"-- last error: {last_error}\n"
        "```"
    )


def call_llm_batch(tasks: list[dict[str, Any]], max_workers: int = 4) -> list[str]:
    """Run multiple LLM calls concurrently and preserve task order."""
    results: list[str | None] = [None] * len(tasks)

    def worker(idx: int, kwargs: dict[str, Any]) -> None:
        try:
            results[idx] = call_llm(**kwargs)
        except Exception as exc:
            results[idx] = f"```lean\n-- Batch worker failed: {exc}\n```"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i, dict(task)) for i, task in enumerate(tasks)]
        concurrent.futures.wait(futures)

    return [result or "```lean\n-- Empty batch result\n```" for result in results]


__all__ = ["call_llm", "call_llm_batch"]
